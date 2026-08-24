"""子智能体模块：具备工具调用能力的真子智能体。

与主智能体（agent_workflow.py）分工不同：主智能体负责全局检索与规划，
子智能体聚焦于主智能体指定的一小组 chunk——它拿到 chunk 清单
（ID+标题+预览）后，自主决定阅读顺序、阅读长度、是否在范围内检索
关键词，完成子任务后调用 finish 提交回答。子智能体拥有独立的
上下文与轮次预算，运行自己的工具循环，因此是真正的多智能体协作。

工具集（受限，只能访问分配的 chunk）：
  - read_chunk:    读取分配范围内某个 chunk 的正文（可限长）
  - search_chunks: 在分配范围内做关键词检索，定位信息位置
  - finish:        提交子任务答案

调用方式（由 agent_workflow.ToolExecutor._dispatch_subagent 调用）：
    from subagent import run_subagent
    result = run_subagent(
        client=client,
        subtask="从分配的资料中找出某事件的具体时间、地点和相关人物",
        allowed_chunks=[
            {"library": "库A", "chunk_id": "zone_001/chunk_000123",
             "heading": "某章节标题", "preview": "正文前 120 字..."},
        ],
        read_chunk=lambda cid, length: "...",
        search_chunks=lambda query, top_k: "...",
        question="用户原始问题",
        temperature=0.3,
        max_rounds=8,
    )
    # result = {"answer": str, "rounds": int, "tool_calls": [...],
    #           "finish_reason": "finish"|"max_rounds"|"error", "error": str}
"""
import json
from typing import Any, Callable, Dict, List


# 子智能体系统提示词（通用 RAG，不绑定具体领域）
_SUBAGENT_SYSTEM = """你是一个子智能体，负责精读主智能体分配给你的一小组资料 chunk，并完成指定的子任务。

可用工具：
- read_chunk(chunk_id, length)：读取分配范围内某个 chunk 的正文
- search_chunks(query, top_k)：在分配范围内做关键词检索，返回命中片段
- finish(answer)：提交子任务的最终回答

工作方式：
1. 先查看分配的 chunk 清单（ID、标题、预览），规划阅读顺序
2. 用 read_chunk 阅读与子任务最相关的 chunk；不确定信息在哪时先用 search_chunks 定位
3. 每轮只调用一个工具，通过 function-calling 通道输出，不要输出任何叙述性文字
4. 信息足够后调用 finish 提交回答；answer 中引用具体内容时标注来源 chunk_id
5. 若分配范围内没有相关信息，如实说明"未找到相关内容"，不要编造
6. 多个 chunk 表述不一致时，分别列出并标注来源"""


# 受限工具集（OpenAI function-calling schema）
SUBAGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_chunk",
            "description": "读取分配给你的某个chunk的正文。chunk_id 必须来自分配清单。",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "chunk 的 ID，必须来自分配清单",
                    },
                    "length": {
                        "type": "integer",
                        "description": "返回文本的最大字符数，默认8000",
                    },
                },
                "required": ["chunk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_chunks",
            "description": "在分配给你的chunk范围内做关键词检索，返回命中片段。用于快速定位信息在哪个chunk，避免逐个全文阅读。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索关键词",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量上限，默认10",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "完成子任务，提交最终回答。",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "子任务的最终回答。引用具体内容时标注来源 chunk_id；范围内没有相关信息时如实说明",
                    },
                },
                "required": ["answer"],
            },
        },
    },
]

# 子智能体单次派遣的轮次预算
DEFAULT_MAX_ROUNDS = 8
# 单条工具结果进入子智能体上下文的最大长度
MAX_TOOL_RESULT_CHARS = 12000


def run_subagent(
    client: Any,
    subtask: str,
    allowed_chunks: List[Dict[str, str]],
    read_chunk: Callable[[str, int], str],
    search_chunks: Callable[[str, int], str],
    question: str = "",
    context_hint: str = "",
    temperature: float = 0.3,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> Dict[str, Any]:
    """执行子智能体（带工具循环）。

    参数：
        client: DeepSeekClient 实例（线程安全，可在任意线程中调用）
        subtask: 子任务描述（要从资料中提取什么信息）
        allowed_chunks: 分配的 chunk 元数据列表，
            每项含 library / chunk_id / heading / preview
        read_chunk: 读 chunk 的回调 read_chunk(chunk_id, length) -> str
            （由调用方实现范围校验与访问记录）
        search_chunks: 范围内检索回调 search_chunks(query, top_k) -> str
        question: 用户原始问题（可选）
        context_hint: 额外上下文提示（可选）
        temperature: LLM 温度
        max_rounds: 子智能体轮次预算

    返回：
        {"answer": str, "rounds": int, "tool_calls": [{"tool","args"},...],
         "finish_reason": "finish"|"max_rounds"|"error", "error": str}
    """
    result: Dict[str, Any] = {
        "answer": "", "rounds": 0, "tool_calls": [],
        "finish_reason": "error", "error": "",
    }
    if not subtask:
        result["error"] = "子任务描述不能为空"
        return result
    if not allowed_chunks:
        result["error"] = "分配的 chunk 列表不能为空"
        return result
    if client is None:
        result["error"] = "子智能体不可用：未配置 DeepSeekClient"
        return result

    # 分配清单：ID + 标题 + 预览，让子智能体自主决定读什么、读多长
    manifest_lines = []
    for c in allowed_chunks:
        disp = f"- {c.get('chunk_id','')}（库: {c.get('library','')}）"
        if c.get("heading"):
            disp += f" · {c['heading']}"
        if c.get("preview"):
            disp += f"\n  预览: {c['preview']}"
        manifest_lines.append(disp)
    manifest = "\n".join(manifest_lines)

    original_q = f"\n用户原始问题：{question}" if question else ""
    hint = f"\n【上下文提示】{context_hint}" if context_hint else ""

    conversation: List[Dict[str, Any]] = [
        {"role": "system", "content": _SUBAGENT_SYSTEM},
        {"role": "user", "content": (
            f"【子任务】{subtask}{original_q}{hint}\n\n"
            f"【分配给你的 chunk 清单】（共 {len(allowed_chunks)} 个，只能访问这些）\n"
            f"{manifest}\n\n"
            "请通过工具阅读资料并完成子任务，完成后调用 finish 提交回答。"
        )},
    ]

    # 懒导入复用主模块的 DSML 兜底解析（避免循环导入；两个模块此时均已加载完成）
    from agent_workflow import _parse_dsml_tool_calls

    last_content = ""
    max_rounds = max(1, int(max_rounds))
    try:
        for round_idx in range(max_rounds):
            result["rounds"] = round_idx + 1
            tool_calls = None
            content_buffer = ""
            for event in client.chat_stream(
                conversation,
                model=client.model,
                temperature=temperature,
                tools=SUBAGENT_TOOLS,
                tool_choice="auto",
                max_tokens=4096,
            ):
                etype = event.get("type")
                if etype == "content":
                    content_buffer += event.get("delta", "")
                elif etype == "tool_calls":
                    tool_calls = event.get("tool_calls", [])

            # DSML 伪标签兜底：模型把工具调用当文本输出时解析为标准 tool_calls
            if not tool_calls and content_buffer:
                parsed = _parse_dsml_tool_calls(content_buffer)
                if parsed:
                    tool_calls = parsed
                    content_buffer = ""

            if not tool_calls:
                # 无工具调用：记录内容并注入纠正提示（与主智能体循环策略一致）
                if content_buffer:
                    last_content = content_buffer
                conversation.append({
                    "role": "assistant",
                    "content": content_buffer or "",
                })
                conversation.append({
                    "role": "user",
                    "content": ("（系统提示：请直接调用工具阅读资料，或调用 finish 提交回答，"
                                "不要输出叙述性文字。）"),
                })
                continue

            conversation.append({
                "role": "assistant",
                "content": content_buffer,
                "tool_calls": tool_calls,
            })

            for tc in tool_calls:
                func = tc.get("function", {}) if isinstance(tc, dict) else {}
                fn_name = func.get("name", "")
                args_str = func.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except (json.JSONDecodeError, ValueError):
                    args = {}
                result["tool_calls"].append({"tool": fn_name, "args": args})

                if fn_name == "finish":
                    answer = (args.get("answer") or content_buffer
                              or last_content or "").strip()
                    result["answer"] = answer
                    result["finish_reason"] = "finish"
                    return result
                elif fn_name == "read_chunk":
                    cid = args.get("chunk_id", "")
                    try:
                        length = int(args.get("length", 8000) or 8000)
                    except (TypeError, ValueError):
                        length = 8000
                    tool_result = read_chunk(cid, length)
                elif fn_name == "search_chunks":
                    query = args.get("query", "")
                    try:
                        top_k = int(args.get("top_k", 10) or 10)
                    except (TypeError, ValueError):
                        top_k = 10
                    tool_result = search_chunks(query, top_k)
                else:
                    tool_result = (f"未知工具: {fn_name}。"
                                   "可用工具：read_chunk / search_chunks / finish")

                if len(tool_result) > MAX_TOOL_RESULT_CHARS:
                    tool_result = (tool_result[:MAX_TOOL_RESULT_CHARS]
                                   + "\n...(结果已截断)")
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": tool_result,
                })

        # 轮次耗尽且未 finish：做一次无工具收尾生成，基于已读内容作答
        result["finish_reason"] = "max_rounds"
        try:
            conversation.append({
                "role": "user",
                "content": ("（系统提示：轮次预算已用完。请基于已读取的资料直接输出"
                            "子任务的回答，引用时标注来源 chunk_id；未读到的信息如实说明。）"),
            })
            resp = client.chat(
                conversation,
                model=client.model,
                temperature=temperature,
                max_tokens=2048,
            )
            answer = resp.get("content", "") if isinstance(resp, dict) else str(resp)
            result["answer"] = (answer or last_content or "").strip()
        except Exception as e:  # noqa: BLE001 - 收尾失败降级用最后内容
            result["answer"] = last_content or f"（子智能体收尾生成失败: {e}）"
        return result

    except Exception as e:  # noqa: BLE001 - 循环中的异常整体兜底
        result["finish_reason"] = "error"
        result["error"] = str(e)
        result["answer"] = result["answer"] or last_content or f"子智能体执行失败: {e}"
        return result

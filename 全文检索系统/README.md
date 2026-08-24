# 本地知识库检索系统

一个面向中文文档的本地知识库检索与智能问答系统。支持多库管理、关键词/语义双通道检索、多种 AI 问答模式（对话/智能体工作流），可纯局域网部署，所有数据落盘本地。

---

## 核心特性

- **多库管理**：按主题划分资料库，库内分 zone 存储，单 zone 字符上限 4000 万，支持跨库并行检索
- **正式身份验证与多用户**：单服务内建账号体系（首个注册用户自动成为管理员）；库级所有权——游客（未登录）只看到并维护公共库（所有游客公用），登录用户看到公共库+自己的库，会话按用户隔离；**支持不同用户创建同名库**（库名在属主内唯一，数据目录按 `libraries/<属主>/` 隔离）
- **前端数据迁移**：登录用户可一键把公共库「复制到我的库」（深拷贝含向量索引）；管理员可转移库所有权（库列表「转移归属」）；CLI 提供 `user` / `library transfer|clone` 命令
- **管理员用户管理**：前端「用户管理」页（仅管理员）——设角色、重置密码、删除用户（含最后管理员保护）；后端提供对应 API
- **并发安全**：同一库的并发导入/恢复由事务级互斥锁串行化，多用户同时导入、导入与语义检索并行均安全
- **多格式导入**：txt / md / markdown / docx / pdf，流式提取+分块+倒排索引+向量索引一条龙
- **双通道检索**：
  - 关键词检索：多关键词共现打分（共现数²×1000 + 窗口紧密度），保留多字短语完整性
  - 语义向量检索：bge-small-zh + Faiss HNSW，支持小 chunk（≤500 字）和大 chunk（父 chunk 最大池化）两级粒度
- **两种 AI 问答模式**：
  - `chat`：纯对话模式，可注入检索结果作为参考资料
  - `agent_workflow`：LLM 自主调用工具循环（list_libraries / search / get_chunk / dispatch_subagent 等）
- **会话上下文管理**：历史问答持久化，支持历史引用 chunk 复用（`list_history_refs` / `filter_history_chunks` 工具）
- **AI 主动数据问题报告**：Agent 工作流中 LLM 自主调用 `report_data_issue` 工具报告资料库内容问题，报告独立管理
- **子智能体派遣**：主智能体可分配 chunk 给独立子智能体精读并提取特定信息
- **导入流程**：实时进度（SSE）、可取消、断点续传、批量状态查询
- **图片型 PDF 检测**：扫描件自动识别并提示 OCR 处理需求
- **PDF 字符数精确估计**：随机采样页提取文本计算平均字数，解决目录/彩页/正文混合场景

---

## 技术栈

| 组件 | 用途 | 说明 |
|------|------|------|
| Python 标准库 `http.server` | Web 服务 | 无额外框架依赖，ThreadingHTTPServer 多线程 |
| `pypdf` / `python-docx` | 文档提取 | 可选依赖，缺失时优雅降级 |
| `sentence-transformers` + `faiss-cpu` | 语义向量 | 默认从本地 `models/bge-small-zh` 加载 bge-small-zh（384 维），无需联网；未装时自动跳过 |
| `jieba` | 标签提取 | 词性过滤提取人物/地名/机构/专有名词 |
| DeepSeek API | 大语言模型 | 支持 V4 Flash（默认）和 V4 Pro |

---

## 快速开始

### 1. 安装依赖

```bash
pip install pypdf python-docx sentence-transformers faiss-cpu jieba
```

### 2. 启动 Web 服务

```bash
python web_api.py
```

默认监听 `0.0.0.0:8000`，允许局域网访问。控制台会输出访问 URL。
如仅本机使用：

```bash
python web_api.py --host 127.0.0.1 --port 8000
```

启动后浏览器打开 `http://127.0.0.1:8000/`，在「设置」页配置 DeepSeek API Key 即可使用智能问答。

### 3. 命令行使用（可选）

```bash
# 创建库
python main.py library create 郎溪县志 --note "郎溪县志全文数据"

# 导入文件（支持目录递归）
python main.py import ./数据目录 --library 郎溪县志

# 关键词检索
python main.py search "农业" --libraries 郎溪县志 --parallel 4 --top 20

# 查看统计
python main.py stats
```

---

## CLI 命令一览

| 子命令 | 作用 | 示例 |
|--------|------|------|
| `library create` | 创建库 | `python main.py library create 库名 --note 备注 [--owner 用户名]` |
| `library list` | 列出所有库（含属主） | `python main.py library list` |
| `library remove` | 删除库（含数据） | `python main.py library remove 库名 -y` |
| `library notes` | 修改库备注 | `python main.py library notes 库名 新备注` |
| `library transfer` | 转移库所有权（数据迁移） | `python main.py library transfer 库名 用户名`（`guest`=设为公共库） |
| `library clone` | 复制库并归属到指定用户 | `python main.py library clone 库名 --to 用户名 [--name 新库名]` |
| `user create` | 创建用户（首个自动为管理员） | `python main.py user create 用户名 --password 密码` |
| `user list` | 列出用户及名下库数 | `python main.py user list` |
| `user remove` | 删除用户 | `python main.py user remove 用户名 -y` |
| `user set-role` | 设置角色（admin/user） | `python main.py user set-role 用户名 admin` |
| `user password` | 修改密码 | `python main.py user password 用户名` |
| `import` | 导入文件 | `python main.py import 文件或目录 --library 库名` |
| `search` | 跨库并行检索 | `python main.py search "关键词" --libraries A B --parallel 4` |
| `verify` | 校验库完整性 | `python main.py verify --library 库名` |
| `recover` | 恢复残留事务 | `python main.py recover --library 库名` |
| `remove` | 删除库内文件 | `python main.py remove --library 库名 --ext txt` |
| `stats` | 查看统计 | `python main.py stats [--library 库名]` |
| `build-index` | 重建索引 | `python main.py build-index --library 库名` |

---

## Web API 概览

### 身份验证与多用户

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/auth/register` | 注册（首个用户自动为管理员；成功后直接登录） |
| POST | `/api/auth/login` | 登录（设置会话 Cookie） |
| POST | `/api/auth/logout` | 登出 |
| GET | `/api/auth/me` | 当前身份（游客/用户/管理员） |
| GET | `/api/auth/users` | 用户列表（仅管理员，含各用户库数量） |
| POST | `/api/auth/users/{name}/role` | 设置角色（仅管理员） |
| POST | `/api/auth/users/{name}/password` | 重置密码（仅管理员，不校验旧密码） |
| DELETE | `/api/auth/users/{name}` | 删除用户（仅管理员；最后管理员不可删） |
| POST | `/api/libraries/{name}/clone` | 把公共库复制到当前用户名下（数据迁移） |
| POST | `/api/libraries/{name}/transfer` | 转移库所有权（仅管理员；`to_owner` 传 `guest` 设为公共库） |

权限规则：未登录=游客，只可见/可写公共库（`owner=guest`）；登录用户可见公共库+自己的库，对自己的库可写；管理员可见/管理全部库，可转移所有权。库列表接口返回 `owner` / `is_public` / `can_edit` 字段。全局设置（`/api/settings`）仅管理员可修改（无任何账号时开放供首次引导）。会话（chat sessions）按用户隔离。

### 库与文件管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/libraries` | 列出所有库 |
| POST | `/api/libraries` | 创建库 |
| PATCH | `/api/libraries/{name}` | 修改备注 |
| DELETE | `/api/libraries/{name}?yes=1` | 删除库 |
| GET | `/api/stats?library=xxx` | 统计信息 |
| GET | `/api/files?library=xxx&search=&ext=&page=1&page_size=50` | 列出库内源文件 |
| GET | `/api/download?file_path=xxx&library=xxx` | 下载源文件 |
| GET | `/api/file-chunks?library=xxx&sha256=xxx` | 获取源文件拼接文本（预览） |
| DELETE | `/api/files?library=xxx&ext=txt` | 按扩展名删除 |
| DELETE | `/api/files?library=xxx&sha=xxx` | 按 SHA 删除 |
| POST | `/api/batch-delete` | 批量删除（流式进度） |
| POST | `/api/delete-all-files` | 清空库内所有文件 |

### 导入

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/import` | 同步导入 |
| POST | `/api/upload-import` | 上传导入（SSE 进度） |
| POST | `/api/import-stream` | 流式导入（SSE 进度，可取消） |
| POST | `/api/upload-stream` | 上传流式导入 |
| POST | `/api/import/cancel` | 取消导入 |
| GET | `/api/import/batch?batch_id=xxx` | 查询批次状态 |

### 检索

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/search?query=xxx&libraries=A,B&parallel=4&top=20` | 关键词检索 |
| POST | `/api/semantic-search` | 语义向量检索 |
| GET | `/api/semantic-status?library=xxx` | 语义索引状态 |
| POST | `/api/semantic-build` | 构建/重建语义索引 |
| GET | `/api/chunks-around?library=xxx&chunk_id=xxx&window=2` | chunk 上下文扩展 |
| GET | `/api/get-chunk?library=xxx&chunk_id=xxx&length=10000` | 获取 chunk 全文 |

### 智能问答

| 方法 | 路径 | 说明 |
|------|------|------|
| GET/POST | `/api/chat-sessions` | 会话管理 |
| GET/POST | `/api/chat-sessions/{id}/context` | 会话额外上下文 |
| GET | `/api/chat-sessions/{id}/export?message_id=xxx` | 导出单条消息为 Markdown |
| POST | `/api/ai-search` | 智能问答（SSE 流式） |
| GET | `/api/suggest-topk` | 推荐 top_k |

### 设置与维护

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/settings` | 读取设置 |
| PUT | `/api/settings` | 更新设置 |
| POST | `/api/test-deepseek` | 测试 API 连通性 |
| POST | `/api/repair-sources` | 修复源文件路径 |
| POST | `/api/backfill-tags-stream` | 批量回填标签（SSE） |
| POST | `/api/verify` | 校验库 |
| POST | `/api/build-index` | 重建索引 |
| POST | `/api/recover` | 恢复事务 |
| GET/POST | `/api/folders` | 文件夹管理 |
| GET/POST | `/api/inquiries` | 质询报告管理 |

---

## 检索语法

智能问答/检索页支持四种查询语法（可混合使用）：

| 语法 | 含义 | 示例 |
|------|------|------|
| `关键词1 关键词2` | 空格分隔，AND 关系 | `刘备 涿县` |
| `(A，B)` | 关键词同义组，OR 关系 | `(死，崩，薨)` |
| `[A，B]` | 语义同义组，向量 OR 召回 | `[战役，交锋]` |
| `{A，B}` | 标题限定组：标题/文件名含组内任一词才保留 | `{水利卷} 灌溉` |

混合示例：`刘备 [死，崩] (涿县，范阳) {蜀书}` 表示同时满足：关键词"刘备" ∩ 语义组"死/崩" ∩ 关键词组"涿县/范阳" ∩ 标题含"蜀书"。

组内分隔符与引号原子：
- 组内分隔符支持中英文逗号、顿号、竖线：`(死，崩)`、`(死、崩)`、`(死|崩)` 等价
- 引号内为原子词项，引号内的分隔符不切分：`("5,40"，"1.5万")` 表示 `5,40` 和 `1.5万` 两个整体词项
- 检索词中无空格的混合串（数字/字母/中文/小数点等）按整体短语精确匹配：`5.40亿立方米` 不会再拆成 `5`/`40`/`亿立方米` 分别匹配；`{水利卷}` 单独使用即为标题检索（按 chunk 标题与文件名匹配）

语义检索阈值：
- `[ ]` 语法触发：相似度 ≥ 0.45（严格）
- 语义检索按钮：相似度 ≥ 0.30（宽松）

---

## AI 模式详解

### chat（纯对话）
不检索，直接对话。可通过 `/api/chat-sessions/{id}/context` 注入检索结果作为参考资料。

### agent_workflow（智能体工作流）
LLM 自主调用工具完成检索+生成，可用工具：

| 工具 | 作用 |
|------|------|
| `list_libraries` | 获取资料库列表 |
| `list_chunk_titles` | 获取库的 chunk 标题列表 |
| `get_chunk` | 获取 chunk 完整文本 |
| `get_neighbors` | 获取相邻 chunk（前后上下文） |
| `search_titles` | 仅在标题中检索 |
| `search` | 正文检索（关键词/多关键词/语义三种模式） |
| `dispatch_subagent` | 派遣子智能体处理指定 chunk |
| `list_history_refs` | 列出本会话历史引用 chunk |
| `filter_history_chunks` | 在历史引用范围内检索 |
| `finish` | 完成检索，开始生成最终答案 |

最大轮数由 `agent_workflow_max_rounds` 控制（默认 15，范围 3-30）。

---

## 配置参数

普通参数保存在 `_settings.json`，敏感字段（如 `deepseek_api_key`）加密后单独保存在 `_secrets.json`，与 `_settings.json` 物理隔离。两者与用户表、会话表一起存放在**用户登录数据目录**（Windows 默认 `C:\Users\<用户名>\biaoshifu`，跨应用共用同一组账号，可用环境变量 `BIAOSHIFU_DIR` 覆盖；见 `userdata.py`）。均可通过 Web 设置页调整，旧版 `_settings.json` 中的明文 API Key 会在首次读取时自动迁移到加密存储。关键参数：

### 规划阶段
| 参数 | 默认 | 说明 |
|------|------|------|
| `plan_max_queries` | 7 | 模型最多规划的查询词数 |
| `plan_max_tokens` | 1000 | 规划 API 响应最大 token 数 |
| `plan_retry` | 2 | 规划 API 失败重试次数 |

### 检索截断
| 参数 | 默认 | 说明 |
|------|------|------|
| `effort_level` | standard | 力度档位：full/boost/standard/economy |
| `max_chunks_per_query` | 9999 | 单查询词引用上限（9999=无限制） |
| `agent_max_rounds` | 3 | Agent 重试轮数 |
| `eval_next_queries_limit` | 3 | 评估补充查询词上限 |

### 精读模式
| 参数 | 默认 | 说明 |
|------|------|------|
| `deepread_snippet_window` | 100 | 小 chunk 窗口：命中位置 ±N 字 |
| `deepread_max_mini_chunks` | 500 | 小 chunk 全局上限 |
| `deepread_expand_max_chars` | 10000 | expand 单次字数上限 |
| `deepread_expand_rounds_per_query` | 2 | 每查询词 expand 轮数 |

### 生成阶段
| 参数 | 默认 | 说明 |
|------|------|------|
| `gen_max_chars_per_chunk` | 800 | 单 chunk 进入 context 字数 |
| `gen_max_total_chars` | 32000 | context 总字数上限 |
| `gen_history_rounds` | 6 | 多轮对话历史轮数 |

### 语义检索
| 参数 | 默认 | 说明 |
|------|------|------|
| `semantic_enabled` | True | 语义通道总开关 |
| `semantic_model_path` | models/bge-small-zh | 本地向量模型目录（存在时优先离线加载，避免联网） |
| `semantic_model_name` | BAAI/bge-small-zh | HuggingFace 模型名（本地路径不存在时回退） |
| `semantic_top_k` | 30 | 单库语义召回条数 |
| `semantic_min_score` | 0.30 | 最低相似度阈值 |
| `semantic_fusion_weight` | 0.5 | 语义/关键词融合权重 |
| `semantic_sub_chunk_size` | 500 | 语义子分块大小（字符数） |
| `vector_queries_per_round` | 3 | 每轮规划的查询词数 |
| `vector_max_rounds` | 3 | 最大规划轮数 |
| `vector_parent_top_k` | 20 | 大 chunk 召回条数 |
| `vector_child_top_k` | 5 | 大 chunk 命中后小 chunk 召回条数 |

### 分块检索
| 参数 | 默认 | 说明 |
|------|------|------|
| `partition_threshold` | 1800 | 触发阈值：≤此值不分块 |
| `partition_max_parts` | 6 | 最大块数 |
| `partition_gradient` | None | 自定义梯度表 [[threshold, parts], ...] |

默认梯度：≤1800→1块，≤3000→2块，≤6000→3块，≤12000→4块，≤24000→5块，>24000→6块。

### 其他
| 参数 | 默认 | 说明 |
|------|------|------|
| `agent_workflow_max_rounds` | 15 | Agent 工作流最大轮数（3-30） |
| `tag_top_k` | 10 | chunk 标签提取数（0=禁用） |
| `compress_threshold_tokens` | 800000 | 上下文压缩阈值（适配 1M token） |

---

## 项目结构

```
search/
├── web_api.py               # Web API 服务入口（http.server，含多用户认证）
├── auth.py                  # 用户/会话管理（注册、登录、PBKDF2 密码哈希、会话 token）
├── userdata.py              # 用户登录数据目录约定（<用户主目录>/biaoshifu）
├── main.py                  # CLI 入口（含 user 管理命令）
├── settings.py              # 设置持久化（_settings.json）
├── chat_store.py            # 会话持久化（_chat_sessions.json）
├── library.py               # 库注册表
├── storage.py               # Zone/ZoneManager 存储
├── importer.py              # 导入编排器（提取→分块→索引→事务）
├── extractor.py             # 文本提取（txt/md/docx/pdf）
├── chunker.py               # 文本分块
├── indexer.py               # 倒排索引
├── searcher.py              # 关键词/多关键词/语义检索
├── ai_search.py             # 通用 AI 检索
├── agent_workflow.py        # Agent 工作流（LLM 自主工具调用）
├── subagent.py              # 子智能体
├── inquiry_store.py         # 数据问题报告存储
├── faiss_index.py           # Faiss 索引封装
├── embedding.py             # 向量嵌入
├── semantic_manager.py      # 语义索引管理
├── deepseek.py              # DeepSeek API 客户端
├── dedup.py                 # 去重索引（SHA256）
├── transaction.py           # 事务管理（staging/commit/rollback）
├── verifier.py              # 完整性校验
├── remover.py               # 文件删除
├── tagger.py                # jieba 标签提取
├── title_extract.py         # 标题提取
├── build.py                 # 打包脚本（PyInstaller）
└── static/
    └── index.html           # 前端单页应用
```

### 数据目录结构

```
{库名}/
├── _sources/                # 源文件副本（溯源用）
├── _semantic/               # 语义索引
│   ├── index.faiss          # 子 chunk 向量索引
│   ├── index_parent.faiss   # 父 chunk 向量索引（最大池化）
│   └── parent_ids.json      # 父 chunk ID 映射
├── zone_001/
│   ├── meta.json            # zone 元数据
│   ├── chunks/              # chunk JSON 文件
│   ├── index/               # zone 级倒排索引
│   └── *.tx.json            # 事务文件
└── zone_002/
    └── ...

_libraries.json              # 库注册表（含 owner 属主字段）
_chat_sessions.json          # 会话存储
_dedup_index.json            # 去重索引
_inquiry_reports/            # 数据问题报告
```

### 用户登录数据目录（`<用户主目录>/biaoshifu`）

用户登录相关数据与库数据分离，统一存放在 `C:\Users\<用户名>\biaoshifu`（Windows），跨应用共用同一组账号密码，可用环境变量 `BIAOSHIFU_DIR` 覆盖：

```
_users.json                  # 用户表（PBKDF2 哈希密码，首个用户为管理员）
_sessions.json               # 登录会话（7 天有效）
_settings.json               # 系统设置（普通参数）
_secrets.json                # 敏感设置（API Key 等，加密存储）
_secret.key                  # 非 Windows 平台 Fernet 密钥（文件权限 0o600）
```

---

## 开发说明

- **修改 Python 代码后需重启 web_api.py**：Python 进程不会自动热重载
- **会话消息持久化**：仅保存用户问题 + 最终回答 + 引用列表，不保存中间过程（思考/检索/工具调用）
- **历史上下文复用**：注入最近 3 轮问答（6 条消息）到 agent 模式，模型可通过 `list_history_refs` / `filter_history_chunks` 工具主动检索历史 chunk
- **跨库查询**：每个库独立走完分块/检索流程后合并结果
- **数据存储区仅支持一级嵌套**：库 → zone，无更深层级

---

## 常见问题

### 导入 PDF 失败提示"图片型扫描件"
PDF 为扫描图片，pypdf 无法提取文本。需先用 OCR 工具（PaddleOCR、ABBYY、Adobe Acrobat）转换为可检索的文本 PDF。

### 语义检索无结果
- 检查库的语义索引状态：`GET /api/semantic-status?library=库名`
- 首次使用需构建索引：`POST /api/semantic-build`
- 大 chunk 模式需要重建索引生成父 chunk 向量

### 服务被占用端口
```bash
python web_api.py --port 8001
```

### 会话消息丢失
会话存储在 `_chat_sessions.json`，如异常中断可能留下空 assistant 消息。系统会在 finally 块自动回填"（生成被中断）"避免会话无法加载。

# ocr-server — 局域网文档 OCR 识别服务

面向地方志（年鉴等）扫描件的局域网 OCR 文档识别服务：上传 PDF / 图片，
识别为可检索 PDF、DOCX、Markdown 等格式输出。支持两种引擎模式：

- `paddle`：本地 PaddleOCR（PP-OCRv6 + PP-StructureV3 版面分析），纯 CPU，无需联网；
- `xfyun`：讯飞云端 OCR（需在讯飞开放平台申请凭证）。

## 快速开始

```bat
:: 1. Python 3.10+，创建虚拟环境并安装依赖
setup.bat

:: 2. 生成本地配置（paddle 模式开箱即用；xfyun 模式填入讯飞凭证）
copy config.example.json config.json

:: 3. 启动
run.bat
```

- 首次以 paddle 模式运行会自动下载 OCR 模型（约 300MB），离线部署请提前准备模型目录。
- 浏览器访问控制台打印的地址（默认 `http://127.0.0.1:8022`），开放注册，
  **首个注册的用户自动成为管理员**。
- 讯飞凭证（`xf` 段的 `app_id` / `api_key` / `api_secret`）也可在管理页面填写，
  即存即生效。`config.json` 含密钥时请勿提交到版本库（已在 `.gitignore` 中）。
- 用户账号数据保存在 `<用户主目录>/biaoshifu/`，本应用的运行数据在 `data/`，
  均不入版本库。

以下为架构与实现细节的技术说明。

> **2026-08 双模式集成**：本软件已合并 ocr-web（讯飞云端 OCR），成为
> **一个软件、两个模式**（`paddle` 本地 / `xfyun` 讯飞云端），共用一套
> **用户名+密码注册制**用户系统。集成说明见下文【双模式与用户系统】。
> 本文档正文仍是原 server-paddle 的并发/内存架构说明。

---

## 双模式与用户系统（2026-08 集成 ocr-web）

### 一、一个软件，两个模式

| 模式 | 值 | 引擎 | 依赖 |
|------|-----|------|------|
| 本地 PaddleOCR | `paddle` | 本机 PaddleOCR 2.7.0.3，纯 CPU | paddlepaddle/paddleocr（重） |
| 讯飞云端 OCR | `xfyun` | 讯飞云服务（standard/llm） | requests/PyMuPDF/docx（轻） |

- **启动选定模式**：`config.json` 的 `mode` 字段（空=首次启动交互式选择）。
  模式一经选定持久化，**重启前固定**——本次启动只注册该模式的路由，
  另一个模式的功能完全不加载、不可用（paddle 模式不 import 任何讯飞代码；
  xfyun 模式不 import paddle，启动快且省内存）。
- 命令行 `--mode paddle|xfyun` 可覆盖（便于脚本/测试），同样只在启动时生效。
- 两个模式共用同一个端口（`config.json` 的 `port`，默认 8070）。

### 二、共用用户系统（注册制）

- 数据文件 `data/users.json`（v2：username + PBKDF2-SHA256 密码哈希，不存明文）。
- **开放注册**：任何访问者可在 `/login` 页注册；**第一个注册的用户自动成为管理员**，
  之后注册的都是普通用户。
- 登录会话为内存态 access_token（7 天有效），重启后需重新登录。
- 旧版 token 制 `users.json`（v1）启动时自动备份为 `users.json.legacy` 并重建空库
  （token 无法换算成密码，旧数据舍弃）。
- 管理员页面 `/admin`（登录页/首页右上角"用户管理"入口）：
  - 启用/禁用用户（禁用后无法登录）
  - 删除用户（连带删除其数据目录）
  - 重置密码
  - 不能删除/禁用自己；不能禁用唯一管理员

### 三、多用户隔离

- **xfyun 模式**：每用户独立目录 `data/<user_id>/`（上传、输出、任务互不可见），
  每用户同时最多 3 个识别任务（`xf.concurrent_limit`）。
- **paddle 模式**：登录门禁 + **任务归属用户**——任务记录 owner（持久化于
  `_tasks_state.json` 与任务 `meta.json`），普通用户只能看到/操作自己的任务；
  管理员可看全部；未登录时代遗留的无 owner 任务仅管理员可见。
- 敏感接口：paddle 的配置修改 `/api/config`（POST）与 `/api/shutdown` 仅管理员；
  `/api/config`（GET）不返回讯飞密钥。

### 四、目录结构（合并后）

```
main.py            # 统一入口：模式交互选择 + 按模式挂载路由
app/
  auth/            # 共用用户系统（users 数据层 / deps 依赖注入 / routes）
  api/             # paddle 模式（原 server-paddle）
  xfyun/           # xfyun 模式（从 ocr-web 移植：ocr_xfyun/exporters/service/routes/static）
  web/             # 共用前端（login.html 登录注册页 / admin.html 用户管理 + paddle 主页）
data/              # 用户库与用户隔离数据（讯飞模式）
output/            # paddle 模式任务输出（保持原状）
ocr-web/           # 已并入 app/xfyun/，此目录不再使用（保留备查）
```

### 五、切换模式

```bash
# 首次启动（config.json 无 mode）：控制台交互选择 1=paddle 2=xfyun
run.bat
# 强制指定（覆盖 config.json）
python main.py --mode xfyun
```

切换模式 = 修改 `config.json` 的 `mode` 后**重启**；运行中无法切换（另一模式路由未注册）。

---

## 一、整体架构

本文重点说明本项目（基于 PaddleOCR 的 PDF OCR 服务）如何实现多线程并发，以及如何避免 PaddleOCR / PaddlePaddle 模型在长文档处理中常见的内存泄漏问题。

---

## 一、整体架构

服务基于 FastAPI + asyncio，对外提供 REST 接口，对内通过三层结构完成 PDF → 拆页 → OCR → 合并 的处理流程：

```
HTTP 请求
   │
   ▼
ConcurrencyManager   ← 第 1 层：asyncio.Semaphore 限制同时运行的任务数
   │
   ▼
TaskManager.run_task ← 第 2 层：ThreadPoolExecutor 把阻塞任务丢到工作线程
   │
   ▼
task_processor       ← 第 3 层：再开一个 ThreadPoolExecutor 按槽位并行处理页
   │
   ▼
SubprocessOCRPool    ← 第 4 层：每个槽位对应一个独立子进程跑 PaddleOCR
```

每一层都解决一个特定问题，缺一不可。

---

## 二、多线程实现

### 2.1 第 1 层：任务级并发控制（asyncio.Semaphore）

文件：[app/api/concurrency.py](app/api/concurrency.py)

PaddleOCR 模型**非线程安全**，多任务同时调用同一实例会崩溃或结果错乱。因此用 `asyncio.Semaphore(max_concurrent)` 限制同时运行的任务数。

核心设计：

- **信号量 + FIFO 队列**：[ConcurrencyManager.\_\_init\_\_](app/api/concurrency.py#L81-L93) 创建容量为 `max_concurrent` 的信号量，并维护一个 `_queue_order` 列表记录排队顺序。
- **槽位栈 `_free_slots`**：每个并发槽位对应一个独立的 PaddleOCR 实例（或子进程），用 LIFO 栈复用，避免多任务共享同一实例。
- **acquire / release**：[acquire](app/api/concurrency.py#L210-L249) 等待信号量后从栈弹出一个槽位号；[release](app/api/concurrency.py#L305-L347) 把槽位压回栈并释放信号量。
- **排队原因可观测**：[get_queue_reason](app/api/concurrency.py#L172-L205) 实时计算排队位置和等待原因，前端可轮询展示。

### 2.2 第 2 层：任务执行线程池（ThreadPoolExecutor）

文件：[app/api/tasks.py](app/api/tasks.py)

PaddleOCR 推理是 CPU 密集型阻塞操作，不能直接跑在 asyncio 事件循环里（会卡死整个服务）。`TaskManager` 用一个常驻线程池承接：

```python
self._executor = ThreadPoolExecutor(
    max_workers=max(concurrency.max_concurrent, 1),
    thread_name_prefix="ocr-worker",
)
```

任务通过 `run_in_executor` 提交到线程池，事件循环本身保持响应，可以继续接收新请求、推送进度。

### 2.3 第 3 层：页级并行（ThreadPoolExecutor）

文件：[app/core/task_processor.py](app/core/task_processor.py)

单个 PDF 任务内部，把页号按 `slots` 数量切分（[_split_pages_to_chunks](app/core/task_processor.py#L527-L537)），再用一个 `ThreadPoolExecutor` 并行处理：

```python
with ThreadPoolExecutor(
    max_workers=n_workers,
    thread_name_prefix=f"ocr-{task_id[:8]}",
) as executor:
    for i, chunk in enumerate(chunks):
        slot = slots[i]
        future = executor.submit(_process_chunk, task_id, chunk, ...)
```

关键点：

- **每个 worker 绑定固定 slot**：页号按 `i % n` 切分，slot 也按索引对应，确保同一时刻一个 slot 只被一个线程使用，PaddleOCR 实例不会并发调用。
- **进度实时上报**：用 `completed_lock + completed_holder` 共享计数器，worker 每完成一页立即回调进度，避免前端长时间显示 0/N。
- **每 10 页重新分配**：[REASSIGN_EVERY_PAGES = 10](app/core/task_processor.py#L34)，主循环每轮处理 `n_workers × 10` 页后重新拉取 pending 列表，避免某个 worker 卡在某页拖累整体。

### 2.4 第 4 层：子进程隔离（SubprocessOCRPool）

文件：[app/providers/subprocess_ocr.py](app/providers/subprocess_ocr.py)

线程池解决了并发调度，但 PaddlePaddle 的 C++ 内存池无法在线程间隔离。因此每个 slot 实际对应一个**独立子进程**，子进程内持有一个 PaddleOCR 实例，通过 stdin/stdout 的 JSON+base64 协议与主进程通信。

```python
class SubprocessOCRPool:
    def __init__(self, pool_size, paddle_config, batch_size=5):
        self._procs = [None] * pool_size          # 每个 slot 一个子进程
        self._locks = [threading.Lock() for _ in range(pool_size)]
        self._restart_locks = [threading.Lock() for _ in range(pool_size)]
```

通信协议见 [app/providers/_worker_ocr.py](app/providers/_worker_ocr.py)：主进程发 `{"type":"ocr","image":...}`，子进程回 `{"type":"result","result":...}`。

### 2.5 关键竞态防护

多线程场景下，取消、超时、重启容易产生竞态，本项目做了细致防护：

| 场景 | 防护措施 | 位置 |
|------|---------|------|
| acquire 等待信号量时被 cancel | try/except 归还许可 | [concurrency.py L235-249](app/api/concurrency.py#L235-L249) |
| 多槽位 acquire 中途取消 | 归还已获取的全部许可和槽位 | [concurrency.py L294-303](app/api/concurrency.py#L294-L303) |
| stall 检测与 ocr() 重试同时重启子进程 | `_restart_locks` 串行化重启 | [subprocess_ocr.py L163-198](app/providers/subprocess_ocr.py#L163-L198) |
| ocr 重试超时与 stall 超时冲突 | 重试次数×超时 < stall_timeout（180s < 300s） | [subprocess_ocr.py L295-299](app/providers/subprocess_ocr.py#L295-L299) |
| 动态调整并发上限 | 重建信号量并保留已占用许可 | [concurrency.py L476-519](app/api/concurrency.py#L476-L519) |

---

## 三、内存泄漏防范（重点：模型部分）

### 3.1 问题根源：PaddlePaddle C++ 内存池不释放

PaddlePaddle 的 `NaiveAllocator` 出于性能考虑，会把推理产生的中间张量缓存在 C++ 内存池中，**不会随 Python 对象 GC 归还给操作系统**。表现为：

- 同进程处理长文档（200+ 页）时，内存从 400MB 单调上涨到 4GB+，每页涨 20-50MB
- 即使 `del` 掉 PaddleOCR 实例并 `gc.collect()`，也只是释放约 500MB，下一页立刻又涨回去甚至更高
- `auto_growth` + `ClearIntermediateTensor` 缓解但仍不足：15 页到 1.4GB

这是 PaddleOCR 在服务端长跑场景下最棘手的问题。本项目通过**多重防线**组合解决。

### 3.2 防线 1：子进程隔离（主防线，最关键）

文件：[app/providers/subprocess_ocr.py](app/providers/subprocess_ocr.py)

**原理**：子进程退出时，操作系统会强制回收该进程的所有内存（包括 PaddlePaddle C++ 内存池缓存的中间张量）。这是唯一能彻底解决 C++ 内存池不释放问题的方案。

**实现**：

- 每个 slot 维护一个长期运行的子进程，处理 `batch_size` 页后自动重启
- [SubprocessOCRPool.ocr](app/providers/subprocess_ocr.py#L286-L356) 中每处理完一页检查计数，达到 `batch_size` 调用 `_restart_proc` 重启：

```python
if self._page_counts[slot] >= self._batch_size:
    self._restart_proc(slot)
```

- [_restart_proc](app/providers/subprocess_ocr.py#L163-L198) 在 kill 旧进程后**显式关闭 stdin/stdout 管道**，让所有阻塞在 `readline()` 上的 reader_thread 立即收到 EOF 退出，避免僵尸线程累积。

**实测数据**（200 页年鉴 PDF，DPI=200）：

| 方案 | 内存表现 |
|------|---------|
| 同进程 | 400MB → 4.2GB，每页涨 20-50MB |
| auto_growth + ClearIntermediateTensor | 基线仍在涨，15 页到 1.4GB |
| **子进程方案（每 5 页重启）** | **主进程稳定在 356MB，零增长** |

### 3.3 防线 2：PaddlePaddle 分配器策略（子进程内）

文件：[app/providers/_worker_ocr.py](app/providers/_worker_ocr.py#L16-L28)

在 `import paddle` **之前**设置环境变量（时机关键，运行后无法修改）：

```python
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
os.environ.setdefault("FLAGS_fraction_of_cpu_memory_to_use", "0.1")
os.environ.setdefault("FLAGS_initial_cpu_memory_in_mb", "128")
```

参数说明：

| 参数 | 作用 | 取值理由 |
|------|------|---------|
| `FLAGS_allocator_strategy=auto_growth` | 张量释放后归还内存给 OS | 默认 `naive` 会缓存不归还，是泄漏根源 |
| `FLAGS_fraction_of_cpu_memory_to_use=0.1` | 单进程内存上限 = 系统内存 × 0.1 | 16GB 系统下单进程上限 1.6GB；旧值 0.3 会让单进程涨到 5GB |
| `FLAGS_initial_cpu_memory_in_mb=128` | 初始预分配 128MB | 降低启动内存峰值 |

### 3.4 防线 3：子进程 RSS 内存阈值主动退出

文件：[app/providers/_worker_ocr.py](app/providers/_worker_ocr.py#L237-L285)

子进程每处理完一页检查自身 RSS 内存，超过 `_MEMORY_LIMIT_MB = 2500` 主动退出：

```python
mem_mb = _get_memory_mb()
if mem_mb > _MEMORY_LIMIT_MB:
    _log(f"内存超限({mem_mb:.0f}MB > {_MEMORY_LIMIT_MB}MB)，主动退出等待重启")
    send({"type": "exit", "reason": "memory_limit", ...})
    break
```

阈值取值 2500MB 的理由：

- 正常页 OCR 内存 < 800MB
- 复杂页（大表格/多栏）可能到 1.5-2GB
- 超过 2.5GB 基本可判定为 paddle 内存泄漏，继续处理只会越涨越高
- 主动退出比被 OOM kill 更优雅，主进程会自动重启子进程

这是一道兜底防线：即使防线 2 的内存上限参数失效，也不会出现单进程 5GB 的情况。

### 3.5 防线 4：主循环每轮重建实例

文件：[app/core/task_processor.py](app/core/task_processor.py#L445-L449)

每轮批处理（`n_workers × 10` 页）结束后，调用 [_rebuild_ocr_instances](app/core/task_processor.py#L614-L671) 重启所有槽位的子进程和版面分析引擎：

```python
# 每轮结束后重建 OCR 实例（释放 paddle 内存池）
_rebuild_ocr_instances(slots, pipeline)
```

该函数同时处理：

- 子进程 OCR：通过 `pipeline._paddle._pool._restart_proc(slot)` 重启
- 版面分析 PPStructure：通过 `pipeline._layout_pool._restart_proc(slot)` 重启
- 同进程模式兜底：调用 `rebuild_ocr_instance(slot)` 重建主进程 PaddleOCR 实例

### 3.6 防线 5：同进程模式的显式内存释放（兜底）

文件：[app/providers/paddle_local.py](app/providers/paddle_local.py#L267-L390)

当子进程模式不可用（如调试、CLI 场景）时，[rebuild_ocr_instance](app/providers/paddle_local.py#L267-L321) 在替换池中引用前显式做四步清理：

1. 调用 `predictor.clear_intermediate_tensor()` 清理中间张量（PaddleInference 官方 API）
2. `del` 旧实例并 `gc.collect()` 触发 Python 层析构
3. 调用 `_release_paddle_global_cache()` 清理全局内存池（`paddle.cuda.empty_cache()` + `core.SetEmptyMemoryPool`）
4. 再次 `gc.collect()` 确保循环引用释放

注释中明确记录了实测结论：仅替换引用而不做上述清理，重建只释放约 500MB，下一页立刻涨回去甚至更高（基线递增）。

### 3.7 防线 6：单页图片内存及时回收

文件：[app/core/task_processor.py](app/core/task_processor.py#L172-L196)

DPI=200 下每页图片约 11.6MB，处理完立即显式释放：

```python
del img
del pix
del page
doc.close()
gc.collect(0)  # 只回收 generation 0，开销 <1ms
```

OCR 结果用完后同样置 None 并 `gc.collect(0)`，避免瞬时累积。

### 3.8 防线汇总

```
┌─────────────────────────────────────────────────────────┐
│  防线 1：子进程隔离（每 5 页重启，OS 强制回收）          │ ← 主防线
├─────────────────────────────────────────────────────────┤
│  防线 2：auto_growth + 内存上限 0.1（子进程内）          │ ← 分配器层
├─────────────────────────────────────────────────────────┤
│  防线 3：RSS > 2500MB 主动退出（子进程自保护）           │ ← 兜底
├─────────────────────────────────────────────────────────┤
│  防线 4：主循环每轮重建所有槽位（task_processor）         │ ← 周期性
├─────────────────────────────────────────────────────────┤
│  防线 5：同进程模式 clear_intermediate_tensor + gc       │ ← 同进程兜底
├─────────────────────────────────────────────────────────┤
│  防线 6：单页图片/结果 del + gc.collect(0)               │ ← Python 层
└─────────────────────────────────────────────────────────┘
```

---

## 四、关键文件索引

| 模块 | 文件 | 职责 |
|------|------|------|
| 并发控制 | [app/api/concurrency.py](app/api/concurrency.py) | 信号量 + FIFO 队列 + 槽位管理 |
| 任务调度 | [app/api/tasks.py](app/api/tasks.py) | 线程池执行 + stall 检测 + 状态持久化 |
| 页处理 | [app/core/task_processor.py](app/core/task_processor.py) | 拆页 / 页级并行 / 合并 |
| 子进程 OCR 池 | [app/providers/subprocess_ocr.py](app/providers/subprocess_ocr.py) | 子进程隔离 + 自动重启 |
| 子进程 Worker | [app/providers/_worker_ocr.py](app/providers/_worker_ocr.py) | PaddleOCR 推理 + 内存自保护 |
| 同进程 Provider | [app/providers/paddle_local.py](app/providers/paddle_local.py) | 实例池 + 显式内存释放 |

---

## 五、性能优化记录（2026-08）

### 5.1 子进程重启改为按真实页数计数（单页 14s → 6.6s）

版面分析路径下，一页会调用多次 `ocr()`（每个文本区域一次 + 整页补充一次）。
旧逻辑按"调用次数"计数（`batch_size=5`），一页 3-10 次调用必然触发重启 →
几乎每页都重启子进程、反复重载模型（5-6s/次），是单页 20-30s 的主因。

修复：
- `SubprocessOCRPool.ocr(..., new_page=False)`：仅 `new_page=True` 的调用
  （每页恰好一次，由 `base.py` 的区域首调用 / 整页补充 / 整页回退标记）计数并
  触发 batch 重启。区域 OCR、`detect_boxes` 不再计数。
- `layout_supplement_ocr`（config.json，默认 true）：false 时跳过版面分析后的
  整页补充 OCR，再省 6-10s/页（代价：PPStructure 漏切区域不补回）。

实测（8 页单页 PDF，模拟版面分析路径 3 次识别/页）：

| 方案 | 平均/页 | 子进程重启 |
|------|--------|-----------|
| 改前（按调用次数） | 14.0s | 4 次 |
| 改后（按页计数） | 12.2s | 1 次 |
| 改后 + `layout_supplement_ocr=false` | 6.6s | 1 次 |

### 5.2 子进程 CPU 线程按并发分配（多并发吞吐 +10%）

旧逻辑不限制子进程线程数（每子进程开满所有核），N 个 OCR 子进程 + N 个
layout 子进程同时满载 → 严重超订。

修复：`subprocess_ocr.py` / `layout.py` 的 `_build_env` 按 pool_size 均分核数
（OCR = cpu_count // pool_size；layout = cpu_count // (2 * pool_size)），
`subprocess_cpu_threads`（config.json，0=自动）可固定覆盖。
单并发时 threads = 核数，行为与旧版一致。

### 5.3 桌面版能力对齐（SLANet 表格 + N 栏排序）

桌面版（OCR-pdf）同步服务器版版面能力：
- `layout.py`：PPStructure `table=True`（SLANet 表格 HTML），模型缺失自动回退
  纯版面分析；`LayoutRegion.html` 字段。
- `base.py`：表格区域 HTML → Markdown 并入输出；`OCRResult.tables` 保留原始
  HTML；`_reorder_by_columns` 从双栏升级为 N 栏。
- 单测：`OCR-pdf/_test_desktop_tables.py`。

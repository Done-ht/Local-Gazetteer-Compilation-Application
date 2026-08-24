# ocr-server — 局域网文档 OCR 识别服务

面向地方志（年鉴等）扫描件的局域网 OCR 文档识别服务：上传 PDF / 图片，
识别为可检索 PDF、DOCX、Markdown、TXT 等格式输出。基于 FastAPI +
PaddleOCR **PP-OCRv6**（纯 CPU 推理），针对长文档（数百页）场景做了
并发调度与内存治理的深度工程化。

## 功能特性

- **双引擎模式**：`paddle` 本地识别（离线可用）/ `xfyun` 讯飞云端识别，二选一启动；
- **PP-OCRv6 + PPStructureV3 版面分析**：多栏/嵌套标题正确排序，表格区域可输出 HTML 结构；
- **多用户注册制**：首个注册用户自动成为管理员，任务按用户隔离；
- **任务级并发 + 页级并行 + 子进程隔离**：四层调度充分利用多核，同时压制内存泄漏；
- **断点续传 / 单页跳过 / 任务级熔断**：长任务中途失败可恢复，引擎故障不会产出整本空白结果；
- **OCR 后处理纠错**：形近字词典、上下文模式等规则化纠错，纠错记录可追溯；
- **打包分发**：PyInstaller 一键打包，模型内置，开箱即用。

## 引擎与依赖

| 组件 | 版本 | 说明 |
|------|------|------|
| paddleocr | 3.7.0 | PP-OCRv6（PPLCNetV4 骨干），兼容指定 PP-OCRv3/v4/v5 |
| paddlepaddle | 3.2.0 | CPU 推理基座 |
| numpy | >=1.24, <2.0 | paddlepaddle 3.x 的 C++ ABI 与 numpy 2.x 不兼容，必须锁定 |
| 版面分析 | PPStructureV3 | PP-DocLayout_plus-L 切割 + SLANet 表格结构 |

**模型档位**（`paddle.ocr_model_tier`，默认 `small`）：

| 档位 | det+rec 体积 | 适用 |
|------|-------------|------|
| tiny | 更小 | 低配机器 / 试运行 |
| small（默认） | ≈30MB | 日常批量，精度/速度均衡 |
| medium | 更大 | 高精度归档需求 |

模型查找顺序：项目本地 `paddleocr_models/<模型名>/` → 打包内置
`_internal/paddleocr_models/` → 首次运行自动联网下载到用户缓存目录
（约 300MB）。离线部署把模型目录放到 `paddleocr_models/` 即可。

## 快速开始

```bat
:: 1. Python 3.10+，创建虚拟环境并安装依赖（阿里云源）
setup.bat

:: 2. 生成本地配置（可选：不创建则首次启动自动生成默认配置）
copy config.example.json config.json

:: 3. 启动
run.bat
```

- 启动时控制台交互选择模式（`1=paddle` / `2=xfyun`）和端口，也可
  `python main.py --mode paddle` 跳过询问；
- 浏览器访问控制台打印的地址（本机 `http://127.0.0.1:<端口>`，引导页含
  局域网地址二维码供同事扫码）；
- 开放注册，**首个注册的用户自动成为管理员**；
- `xfyun` 模式需讯飞开放平台凭证：填入 `config.json` 的 `xf` 段，或在管理
  页面「讯飞 OCR 设置」填写（即存即生效）。**凭证不要提交版本库**
  （`.gitignore` 已忽略 `config.json`，仓库内只提供 `config.example.json`）。

数据存放：账号数据在 `<用户主目录>/biaoshifu/`（可用环境变量
`BIAOSHIFU_DIR` 重定向，可与同系列"全文检索系统"共用一套账号）；上传/
输出/任务等运行数据在应用目录 `data/<user_id>/` 与 `output/`。

## 配置说明（config.json 主要字段）

| 字段 | 默认 | 说明 |
|------|------|------|
| `mode` | `""` | `paddle` / `xfyun`；空 = 每次启动交互选择，选定后写入 |
| `port` | `8070` | 监听端口（启动时还可临时修改） |
| `render_dpi` | `200` | PDF 页渲染 DPI，越高越清晰越慢 |
| `max_concurrent` | `3` | 同时处理的任务数（每任务一个 OCR 槽位） |
| `output_format` | `original` | `original` / `searchable_pdf` / `markdown` / `json` / `txt` |
| `paddle.ocr_version` | `PP-OCRv6` | 亦可指定 `PP-OCRv5` / `PP-OCRv4` / `PP-OCRv3` 回退 |
| `paddle.ocr_model_tier` | `small` | 模型档位 `tiny` / `small` / `medium` |
| `paddle.enable_layout` | `true` | PPStructureV3 版面分析（多栏排序必需） |
| `paddle.use_table_recognition` | `false` | SLANet 表格结构识别，开启约增加 30-40s/页 |
| `layout_supplement_ocr` | `true` | 版面切割后对整页做一次补充 OCR，兜住漏切区域；关闭更快 |
| `subprocess_cpu_threads` | `0` | 每个 OCR 子进程的 CPU 线程数；0 = 按并发数自动均分核数 |
| `xf.*` | — | 讯飞凭证与限流参数（仅 xfyun 模式使用） |

## 架构设计

### 四层并发调度

```
HTTP 请求
   │
   ▼
ConcurrencyManager   ← 第 1 层：asyncio.Semaphore 限制同时运行的任务数
   │
   ▼
TaskManager.run_task ← 第 2 层：ThreadPoolExecutor 把阻塞任务移出事件循环
   │
   ▼
task_processor       ← 第 3 层：再开一个 ThreadPoolExecutor 按槽位并行处理页
   │
   ▼
SubprocessOCRPool    ← 第 4 层：每个槽位一个独立子进程跑 PaddleOCR 推理
```

1. **任务级信号量**（[app/api/concurrency.py](app/api/concurrency.py)）：
   PaddleOCR 实例非线程安全，用 `asyncio.Semaphore(max_concurrent)` + FIFO
   排队限制并发；每个任务绑定独立槽位号（LIFO 栈复用），排队位置实时可查
   （前端展示等待原因）。
2. **任务线程池**（[app/api/tasks.py](app/api/tasks.py)）：CPU 密集推理不阻塞
   asyncio 事件循环，服务持续可响应。
3. **页级并行**（[app/core/task_processor.py](app/core/task_processor.py)）：
   单任务内部把页号按槽位数切分并行处理；每 `REASSIGN_EVERY_PAGES=10` 页
   重新拉取待处理页，避免慢页拖累整体；进度逐页回调上报。
4. **子进程隔离**（[app/providers/subprocess_ocr.py](app/providers/subprocess_ocr.py)）：
   每槽位一个常驻子进程持有 PaddleOCR 实例，stdin/stdout JSON 协议通信
   （见 [app/providers/_worker_ocr.py](app/providers/_worker_ocr.py)）；
   按**真实页数**（而非接口调用次数）累计满 `batch_size=5` 页后自动重启
   该子进程。取消/超时/重启的竞态均有对应防护（许可归还、重启串行化、
   重试超时 90s×2 < stall 检测 300s 等）。

### 内存治理（六道防线）

PaddlePaddle 的 C++ 内存池不会随 Python GC 归还操作系统，长文档场景下
同进程内存单调上涨。本项目组合以下防线：

```
防线 1  子进程隔离 + 按页数批次重启        ← 主防线，OS 强制回收
防线 2  分配器策略（子进程内，import 前设置）← 官方推荐方案
防线 3  子进程 RSS > 2500MB 主动退出       ← 自保护兜底
防线 4  跨任务累计 15 页全量重建           ← 周期性回收，兼顾冷启动开销
防线 5  同进程模式显式释放中间张量 + gc    ← 调试/CLI 兜底
防线 6  单页图片/结果用后即删 + gc.collect(0)
```

- **防线 2 细节**（[_worker_ocr.py](app/providers/_worker_ocr.py)）：采用
  PaddleOCR 官方推荐的 `naive_best_fit + eager_delete` 组合
  （`FLAGS_eager_delete_scope=True`、`FLAGS_eager_delete_tensor_gb=0.0` 等），
  并以 `FLAGS_fraction_of_cpu_memory_to_use=0.1` 限制单实例内存上限。
  实测 CPU 场景下该组合优于 `auto_growth`（后者归还策略偏弱，15 页仍涨至
  1.4GB）。
- **防线 4 权衡**：旧版每个任务结束都全量重启所有 worker，导致小任务也要付
  约 60s 冷启动税；现改为跨任务累计满 `REBUILD_AFTER_PAGES=15` 页才全量
  重建一次，中间小任务复用热 worker。
- 实测基线（200 页年鉴 PDF，DPI=200）：纯子进程方案下主进程稳定在约
  356MB 零增长（无子进程隔离时同进程会从 400MB 涨至 4GB+）。

### 长任务可靠性

- **断点续传**：拆页产物持久化，中断重跑自动跳过已完成页；
- **单页熔断**：同一页失败达 `MAX_PAGE_FAILS=3` 次即跳过（合并阶段插空白页），
  避免无限重试卡死任务；
- **任务级熔断**：尝试页数 ≥8 且失败率 >50% 判定引擎级故障，直接中止任务，
  杜绝"整本书静默输出空白"。

## 目录结构

```
main.py                  # 统一入口：模式/端口交互 + 按模式挂载路由
paddleocr_rt_hook.py     # 打包运行时钩子（修正 frozen 环境资源定位）
requirements.txt         # 依赖清单（含版本约束原因注释）
setup.bat / run.bat      # 环境搭建 / 启动
build.bat + main.spec    # PyInstaller 打包（模型一并打入 _internal/paddleocr_models/）
config.example.json      # 配置模板（复制为 config.json 使用；真实密钥不入库）
app/
  auth/                  # 共享用户系统（users 数据层 / deps 鉴权 / routes 管理）
  api/                   # paddle 模式 REST：concurrency / tasks / routes(配置、关机等)
  core/
    pipeline.py          # 识别流水线编排（OCR + 版面 + 过滤）
    task_processor.py    # 任务执行：拆页 / 页级并行 / 合并 / 重建策略
    layout.py            # PPStructureV3 版面分析封装
    filter.py            # 双层漏斗过滤（纯图页快速跳过）
    text_corrector.py    # 中文 OCR 规则化纠错（形近字/上下文/数字校验）
    document/            # 文档解析与输出：pdf/image/docx handler
  providers/             # OCR 引擎层：subprocess_ocr 池 / paddle_local 同进程 /
                          # _worker_ocr、_worker_layout 子进程脚本
  utils/                 # config / output(txt/md/json/可搜索PDF) / 进度日志 / 任务目录
  web/                   # 共用前端（登录注册 / 主界面 / 用户管理）
  xfyun/                 # xfyun 模式（service / exporters(docx,pdf) / routes / static）
data/                    # 运行时生成：各用户上传与输出（不入库）
output/                  # 运行时生成：任务状态与日志（不入库）
```

## 权限与安全边界

- `/api/config` 写接口与 `/api/shutdown` 仅管理员；读接口剔除讯飞凭证字段；
- 任务归属持久化（owner），普通用户只能看到/操作自己的任务，管理员可见全部，
  未登录时代的无 owner 旧任务仅管理员可见；
- 密码 PBKDF2-HMAC-SHA256（20 万次迭代）加盐哈希存储，绝不明文落盘；
  会话 token 服务端生成并持久化（7 天有效，重启不丢登录态）；
- 管理员不可删除/禁用自己，且系统始终保证至少一名可用管理员。

## 打包分发

```bat
build.bat    :: PyInstaller 打包，产物在 dist\server-paddle\
```

- 打包会把 `paddleocr_models/` 下的模型一并收入 `_internal/paddleocr_models/`，
  目标机器无需联网下载模型；
- 打包版配置文件读取 exe 同级目录的 `config.json`，账号数据仍在用户主目录。

## 二次开发提示

- OCR 结果统一序列化为 `[[box, [text, conf]], ...]`，主进程解析与引擎解耦，
  替换引擎只需实现 provider 接口；
- 版面分析只负责切割 + 表格结构，文本识别一律走 Provider，两条链路互不耦合；
- `ocr_version` 保留了 v3/v4/v5 分支作为回退（各 2 行分派代码），切换只需改配置。

# PRTS Core 第二阶段集成候选

这是供前端团队和集成 agent 使用的后端核心包。包含 Python 可运行参考、实际模型权重、C++ 核心接口、Swift Package 源码、录像、截图、原理图和测试证据。**Apple 源码未在 Mac/Xcode/iPhone 编译执行；此包不是已验收的 iOS 二进制。**

先看 `docs/PRTS_CORE_STAGE2_PROGRESS.md` 了解现状，`docs/PRINCIPLES.md` 说明处理过程，四张可编辑原理图在 `docs/diagrams/`，包括新增叫号证据流程。文件 SHA256 位于 `BUNDLE.json`，各模型来源、量化配方和许可随模型清单及 `licenses/stage2/` 提供。

## 在 Windows 复现

Python 3.11、64 位 Windows，ffmpeg 需在 PATH。此参考环境不安装 Torch 或 CUDA。使用 CPU：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-core-cpu.txt
.venv/Scripts/python.exe scripts/run_bundle_replay.py --verify-only --output outputs/check
.venv/Scripts/python.exe scripts/run_bundle_replay.py --case bus --output outputs/my-bus --provider cpu --tts
```

若要使用 Windows DirectML，在另一个环境安装 `requirements-core-dml.txt`，再把 provider 改为 `dml`。两个 ORT 包不要装到同一环境。`--tts` 需要 Windows 离线中文 Huihui 音色，不自动打开扬声器；没有该音色时先不传该选项，事件中的文字仍由实际模型产生。

还可运行 `--case notice`（真实录音读告示）和 `--case citybus20`（20 路滚动电子牌、脚本文字请求）。全部相机输入来自已有录像，按原速送入模型；不是现场摄像头采集。CPU 地面分割可能超过引导新鲜度，此时系统会返回等待，不能把离线录像流畅度当成 CPU 实时导航性能。重复运行请使用不同输出目录。

主输出是 `events.jsonl`、`summary.json`、标注录像、截图和可选的运行时 TTS WAV。接口是进程内调用，前端输入相机/连续 PCM/定位，消费事件。API 契约与时钟坐标在 `docs/INTERFACE_SPEC.md`。

新增 `--case queue`：三张真实叫号照片按明确时序拼接，四段合成用户 PCM，加两条标明的文字命令；验证窗口号反例、215 / 494 / B003 提醒及修改取消。它是模型与交互回放，不是真实排队过程录像或现场 ASR 精度测试。对应原理图为 `04-queue-evidence.svg`，首次误报与未解决的 LED 栏目均保留在 `docs/QUEUE_VALIDATION.md`。

## 让 agent 接入 Apple 工程

把整个包交给团队的 agent，要求它先阅读 `docs/APPLE_INTEGRATION.md`，在 Mac 执行 Foundation 对照和 `scripts/build_apple_native.sh`，然后把 `apple/PRTSCore` 作为本地 Swift Package 加入现有 App。模型路径由 `CoreModelFiles` 显式传入，Windows DLL 不用于 iOS。

入口 `PRTSCoreSession` 已接好模型、任务、地图和输出事件；前端负责相机/麦克风、采集时间映射、回声消除、实际播报状态与界面。包内不含地图 Key，团队通过 `AMapService` 注入自己的凭据。新目的地检索和路线规划需要地图服务联网，本地感知/问答/等待不需要联网。

Apple 使用系统 Vision OCR，与桌面 PP-OCRv6 不同；Swift 路径还没有 Python 光流注册。必须按随包案例做 Apple 实测，记录真实失败，不用空回调代替模型。LiDAR/AR 位姿与深度接口已预留，但本轮没有世界空间 AR 注册效果。

## 可以展示的证据

| 路径 | 能说明什么 |
|---|---|
| `outputs/stage2/demos/citybus20-dml-q8-v2-timeline/demo.mp4` | 当前 Q8 视觉组合，真实 20 路滚动电子牌；脚本文字请求，60/60 帧，5.5 秒产生目标确认 |
| `outputs/stage2/demos/bus-dml-q8-v2-timeline/demo.mp4` | 当前 Q8 组合与真实用户录音，61/61 帧，目标确认时证据年龄 3.938 秒 |
| `outputs/stage2/demos/bus-dml-v5-timeline/demo.mp4` | F16 视觉基线、真实毕设输入和真实用户语音、运行中生成 TTS；按可用时间重建播放，不是扬声器实录 |
| `outputs/stage2/demos/notice-timeline/demo.mp4` | 真实录音、告示 OCR 与文字答复，保留否定条件 |
| `outputs/stage2/perception/continuous-turn/source-time.labelled.mp4` | 相同胸前视频上的旧/新感知路径对照；源时间播放，不代表推理速度 |
| `outputs/stage2/perception/continuous-turn/processing-time.labelled.mp4` | 上述对照按实测处理时间展开 |
| `outputs/stage2/maps/route-replay/route-demo.mp4` | 高德真实路线，627 个明确仿真的位置，地点确认、转弯、到达后等车；不是现场 GPS 或地图与相机 AR 对齐 |
| `outputs/stage2/bus-generalization/holdout-v3-review/review-sheet.png` | 首次保留测试中的成功与失败，包括 101X 漏报 |

各演示的模型版本不同，以其原始清单为准，不能把 F16 演示当作 Q8 的运行结果。包的当前权重组合见 `BUNDLE.json`，量化质量与 20 分钟预算见 `docs/RUNTIME_RESULTS.md`。

已知未达成项包括 101X 的远距/小字漏识别、叫号 ASR 字母丢失、两种 LED 栏目未确认、真实地铁等待、信号灯方向与灯色绑定、同一路线现场空间注册及 Apple 实测。通过七个公交开发用例不等于对所有线路可靠；开发数据与首次保留数据分开报告。原失败目录与报告随包保留。

`docs/STAGE2_THIRD_PARTY_NOTICES.md` 说明模型和数据各自的许可；Mask2Former 权重有非商业条款。毕设素材仅用于本团队研究与集成，包未向外发布。

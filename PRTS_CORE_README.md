> 当前多模态核心交付入口：[BRAIN_PROTOTYPE](docs/BRAIN_PROTOTYPE.md)、[最终验证](docs/FINAL_VALIDATION.md)、[ZIP 使用说明](docs/BRAIN_BUNDLE_README.md)。下文保留早期原型记录，不能用于替代新包的模型与资源结论。

# PRTS Core 离线后端 Demo

本文保留第一阶段的运行方式与限制。第二阶段的连续后端、真实地图、模型量化和 Apple 接口请从 [当前实施记录](docs/PRTS_CORE_STAGE2_PROGRESS.md) 与 [Apple 接入说明](docs/APPLE_INTEGRATION.md) 开始；下文串行调度和无路线规划的描述仅属于第一阶段。

这是一个可运行的 Python 后端：相机帧或视频 → 人行道分割与障碍检测 → 局部候选通道；用户录音 → 中文识别 → 导航、停止、场景问答或持续等待任务。输出 JSONL、调试视频及可选中文答复 WAV。没有新增前端，原有 `floor_nav.py` 和浏览器原型继续保留。

**当前验收范围是桌面原型，不是可独立替代盲杖的出行产品。** 路径位于图像坐标系中，未标定人体宽度、米制距离、落差或安全过街条件；不提供目的地路线规划。分割漏检会暂停，检测误判也会导致不必要的停止。详细实测与失败案例见 [验证记录](docs/PRTS_CORE_RESULTS.md)。

## 在本机直接运行

从仓库根目录使用已准备好的环境和本地权重：

```powershell
.venv-research\Scripts\python.exe -m prts_core.demo --scenario outputs\cases\sidewalk.json --output outputs\try-sidewalk
.venv-research\Scripts\python.exe -m prts_core.demo --scenario outputs\cases\labelled.json --output outputs\try-obstacles
.venv-research\Scripts\python.exe -m prts_core.demo --scenario outputs\cases\voice_navigation.json --output outputs\try-voice --tts
```

第一条观察普通城市人行道，第二条观察障碍改变路径，第三条将语音启动、场景问题、语音停止串起来。第三条的语音由本地系统语音合成，仅用于集成测试；另有用户提供的真人录音验证。

```powershell
.venv-research\Scripts\python.exe -m prts_core.demo --scenario outputs\cases\notice.json --output outputs\try-notice --tts
.venv-research\Scripts\python.exe -m prts_core.demo --scenario outputs\cases\bus.json --output outputs\try-bus --tts
.venv-research\Scripts\python.exe -m prts_core.demo --scenario outputs\cases\numbers.json --output outputs\try-announcement --tts
.venv-research\Scripts\python.exe -m prts_core.demo --scenario outputs\cases\numbers_visual.json --output outputs\try-screen --tts
```

告示与公交案例引用用户本机原始文件。仓库不包含私人视频、录音或权重；上述个人案例清单只存放在忽略提交的 `outputs/` 中。叫号屏幕和广播均为合成素材，不能当作真实医院或车站准确率。

`--tts` 在回放结束后把回复导出到 `speech/*.wav`，不自动播放。它依赖已安装的离线中文系统语音，本机使用 Microsoft Huihui。JSONL 事件是后续客户端实时播报的接口。

## 从零准备

Python 3.11 为已测环境。安装适合本机的 [PyTorch CUDA 构建](https://pytorch.org/get-started/locally/)；本机 RTX 4070 SUPER 使用 CUDA。CPU 可用 `--device cpu`，但没有完成 CPU 全场景性能验收。

```powershell
python -m venv .venv-research
.venv-research\Scripts\python.exe -m pip install -r requirements-offline.txt
.venv-research\Scripts\python.exe scripts\download_models.py brain asr-medium seg-sidewalk detector
.venv-research\Scripts\python.exe scripts\download_demo_data.py
.venv-research\Scripts\python.exe scripts\prepare_demo_cases.py
```

下载步骤需要联网，推理只读取本地文件。模型下载脚本固定上游 revision，并记录文件 SHA256。`requirements-offline-tested.txt` 记录本机实际包版本，包含 CUDA 构建标记，不能视为所有系统的通用安装锁文件。RapidOCR 固定为 3.9.2，运行前检查其三个本地 ONNX 权重；缺文件时直接报错。

只生成公开视频清单、跳过 Windows 合成语音时，加 `scripts/prepare_demo_cases.py --skip-synthetic`。准备私人回放时使用 `--private-dir <原始视频所在文件夹>`，需要 `ffmpeg` 在 PATH 中；录音片段写入 `outputs/cases/`，原视频不移动。

## 接入自己的输入

创建 JSON 文件，路径可以是绝对路径，或相对于仓库根目录：

```json
{
  "source": "D:/samples/walk.mp4",
  "start": 0,
  "end": 20,
  "interval": 0.5,
  "navigate": false,
  "commands": [
    {"at": 0, "audio": "D:/samples/start.wav"},
    {"at": 8, "text": "看看前面有什么障碍"},
    {"at": 15, "text": "停止导航"}
  ]
}
```

运行 `python -m prts_core.demo --scenario <文件> --output <输出文件夹>`。`interval` 是媒体采样间隔，不是性能目标。`source` 也支持图片或逐帧清单：`{"frames":[{"file":"frames/000000.png","time_s":0}]}`，清单内图片路径相对于清单文件。

等待任务可同时接收视觉和环境录音：

```json
{
  "source": "D:/samples/station.mp4",
  "interval": 0.5,
  "commands": [{"at": 0, "text": "等十七路公交来了提醒我"}],
  "ambient": [{"at": 8, "start": 5, "audio": "D:/samples/announcement.wav"}]
}
```

`at` 表示该段录音何时可交给识别器，`start` 表示录音起点。任务开始之前的旧广播不会触发新任务。公交线路必须在检测到的车辆中读出，或者在明确到站广播中出现；叫号需号码精确匹配且有叫号语义。等待可修改、取消，成功后只提醒一次。当前支持一项等待任务；交通灯持续等待会明确返回 `unsupported_task`。

摄像头入口：

```powershell
python -m prts_core.demo --camera 0 --output outputs\live
```

终端输入文字命令，输入 `/mic` 录制 5 秒用户指令，输入 `/quit` 结束。文件回放已经实测；本轮未完成实体摄像头、麦克风的现场验收。直播入口尚未连续监听环境广播，广播输入通过回放清单或 `Engine.process(..., ambient=...)` 接入。

## 输出及内部边界

- `events.jsonl`：转写、意图来源、回答、导航状态、目标事件与逐帧观测。区分 `semantic_rules`、本地语言模型、OCR 和视觉模型来源。
- `summary.json`：状态统计、分阶段处理时间、采样 RSS、PyTorch 显存峰值及完整问答记录。模型初次加载、语音处理和逐帧导航分别看待。
- `annotated.mp4`、`preview.jpg`：绿色为分割候选，橙色为被检测框剔除的部分，紫色为无检测的基线路径，蓝色为过滤后的路径。时间标注是源媒体时间。
- `question_*.jpg`、`target_*.jpg`：问答和到达提醒的实际取证帧；`speech/manifest.json` 将答复 WAV 对应回事件。

`CANDIDATE` 表示观察到候选通道；`DETOUR` 要求同一近场起点下障碍改变了可达路径；`STOP` 表示原本候选通道被检测框阻断；`WAIT` 表示分割不连续、镜头移动或帧过期。状态不是安全通行保证。

感知默认使用 SegFormer-B4 Sidewalk 与 YOLO11n。语音使用 Whisper medium INT8 CPU，本地 VLM 为 MiniCPM-V-4.6 BF16 CUDA；明确控制语义先规则路由，复杂表达再由模型分类。读牌直接整理 OCR 原文，保留主体与否定；普通场景问题结合当前图像和当前检测，不把无关 OCR 文字当作挡路证据。

相机采集仅保留最新帧，任务状态与最近 12 次场景摘要形成短期记忆。等待随新观测持续检查，目标出现才产生事件。当前语言推理与导航是串行的：处理用户命令时明确暂停引导，问答取新帧，恢复时再次取最新帧，超过 1.5 秒的导航帧返回 WAIT。**这属于系统层持续观察和任务编排，不是模型原生全双工，也未实现说话打断和边说边听。**

## 验证和实验

```powershell
python -m unittest discover -s tests -p "test_offline_core.py" -v
npm test
python scripts/offline_smoke.py --scenario outputs/cases/road.json --output outputs/offline-check --tts
```

离线冒烟测试禁用当前 Python 进程的 socket 连接，再运行真实 ASR、OCR、视觉问答和 TTS；这是进程级检查，不等同操作系统断网或网络抓包。

可选训练实验：

```powershell
python scripts/download_models.py seg
python scripts/download_demo_data.py --include-training
python scripts/train_sanpo.py
python scripts/evaluate_segmentation.py
```

仅使用官方标为 `HUMAN_ANNOTATED` 的帧，按会话分开训练、开发与测试。开发集选型先约束车道误纳入，再比较 IoU。独立测试发现公开标签异常，自训练模型保留实验用途，未替换默认 B4；不得把演示帧、机器掩码或训练样本当作独立人工验证。

## 模型许可与后续平台

公开回放数据来自 [SANPO](https://github.com/google-research-datasets/sanpo_dataset)，适用 CC BY 4.0；数据来自真实步行视频，部分掩码为机器传播结果。权重条款需分别看待：[NVIDIA SegFormer](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024) 有研究/评估限制，[Sidewalk 权重](https://huggingface.co/nickmuchi/segformer-b4-finetuned-segments-sidewalk) 的模型卡标注不能抹去上游限制，[Ultralytics](https://www.ultralytics.com/license) 为 AGPL/商业许可选择。没有将这些权重重新声明为本仓库许可证。

Windows 上完成的功能与测量不能外推为 iOS 成果。本轮未用 Xcode、Core ML 或真机完成部署；Apple 的权重转换、量化质量、峰值工作集、热状态与后台音视频权限仍需独立验证。[原计划](docs/PRTS_CORE_PLAN.md) 保留平台调研及历史阶段记录。模型不固定绑定 O 系列：V 模型加 ASR、OCR、任务状态和事件调度是当前可运行组合，替换模型应以相同输入上的实际效果为依据。

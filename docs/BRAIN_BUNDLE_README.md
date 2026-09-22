# PRTS Core 多模态快速原型（自由前进修订版）

本次修改与新公交录像以 `docs/REVISION_FREE_FORWARD.md` 为准。自由前进修改通过测试；20 路连续录像成功提醒，20A 反例未误提醒。912 录像漏提醒，综合回放 101 因推理过慢导致证据超时而未提醒；综合三目标验收未通过，详见 `BUNDLE.json` 的 `replay_acceptance`。

这是含本地权重、原生运行时、Python 后端、Apple 接口源码和可追溯演示的交接包。等待公交、叫号与复杂视觉问答由实际 Qwen3-VL-4B 多模态模型完成；Mask2Former 与 YOLO 持续提供地面、障碍物和图像坐标候选路径。文字回复交给系统 TTS，提示音由事件驱动。没有前端界面。

本包为已运行的桌面快速原型。Apple 源码仍须团队在 Mac 构建与真机验证；整套应用的 8 GB、远处小标识、移动目标提醒延迟和世界空间 AR 尚未全部达标。当前实测与失败案例见 `docs/FINAL_VALIDATION.md`，不要套用历史 OCR 方案的资源指标。

## 先看实际效果

- `outputs/stage2/demos/free-forward-final-timeline/demo.mp4`：连续相机/PCM 输入、局部感知、等待换目标、取消、视觉问答与 TTS。使用真实公共照片和胸前录像拼接，脚本命令及合成用户语音已注明；不是一次实地行程。
- `outputs/stage2/streaming/free-forward-final/events.jsonl`：每个结果对应的实际模型输入、原始输出、证据时间、任务版本与播报请求；`summary.json` 含整套内存采样结果。
- `outputs/stage2/maps/route-replay/route-demo.mp4`：真实高德步行路线响应与模拟 GPS 回放，包含目的地候选、确认、转弯与到达。没有虚构街景或 AR 注册。
- `outputs/stage2/perception/continuous-turn/`：连续真实道路视频的地面分割、障碍区域与候选路径。
- `docs/diagrams/05-brain-cerebellum.svg` 与 PNG：比赛讲解用大脑、小脑、任务状态和输出链路；旧编号图描述历史试验。

新的连续公交录像：`outputs/stage2/demos/citybus20-brain-v3-timeline/demo.mp4`（等20）、`citybus20-other-v3-timeline/demo.mp4`（同段录像等20A反例）、`private-bus-brain-v2-timeline/demo.mp4`（用户912录像）。这些使用新大脑、原视频1×输入，不是照片循环。

## Windows 本地复现

在解压根目录创建 Python 3.11 环境。已测 DirectML；不依赖 CUDA，纯 CPU 也可调用，但地面处理可能慢到只返回 WAIT。不要混装两个 onnxruntime 包。

```powershell
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-core-dml.txt
.venv/Scripts/python.exe scripts/run_bundle_replay.py --verify-only --output outputs/verify
.venv/Scripts/python.exe scripts/run_bundle_replay.py --case brain --provider dml --tts --output outputs/replay-new
```

录像合成另需 PATH 中的 FFmpeg；Windows TTS 需要系统中文 Huihui 语音。首次预解码会花几分钟，计时回放从模型预热完成后开始。程序生成 WAV，不访问物理扬声器。`--case citybus20-brain`、`--case citybus20-other`、`--case private-bus-brain` 可重放本次三组连续公交；`--case bus` 和 `--case notice` 使用用户提供的毕设录像与语音；这些短视频不保证较慢 CPU 能在目标离开前完成新的大脑推理。

`BUNDLE.json` 指定实际选用的运行时和模型参数，`MODEL_MANIFEST.json` 提供权重散列。`run_bundle_replay.py` 先逐文件校验再运行。原始测量日志保留研究机路径作为溯源；`examples/` 与上述运行入口已改成解压目录内路径，不需要原工作树。

## 交给 Apple 集成 agent

按顺序阅读 `docs/BRAIN_PROTOTYPE.md`、`docs/INTERFACE_SPEC.md`、`docs/APPLE_INTEGRATION.md` 和最终验证报告。将 `apple/PRTSCore` 接入现有 App，构建 `scripts/build_apple_native.sh` 固定提交的 XCFramework。Windows DLL 和 Python 虚拟环境不能嵌入 iOS。

输入为转正相机帧、16 kHz 单声道 PCM、可选定位与朝向；输出为场景/区域/路径、任务事件、答复文字、TTS 与提示音。保留真实时间戳、取消与任务版本、播放反馈。LiDAR、深度及 AR 位姿已有输入契约，当前不消费这些字段生成世界坐标路径。

感知、ASR 和大脑可本地运行；新高德路线查询需要网络和团队自己的 Key。演示回放无需 Key，包内不含本次临时凭据。当前没有原生音频生成模型和真实地铁等待实证。先复现桌面冻结用例，再比较 Apple Vision OCR、CPU/Metal/CoreML 的输出、峰值与持续运行表现。

## 数据与版本

公共素材的出处与许可在相应 `sources.json`、素材清单和 `licenses/` 中。用户毕设视频随包作为用户提供素材保留。模型选择没有在这几段演示上训练权重；既有开发素材及重复比较不能被称为独立盲测。原始失败实验随包保留，用于确定下一阶段的优先事项。

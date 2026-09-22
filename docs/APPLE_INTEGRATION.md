> 2026-09-22 后续修订：[自由前进与连续公交演示](REVISION_FREE_FORWARD.md) 为最新行为与验证入口；下文先前测试数据保留其原版本范围。

# 给 Apple 前端团队及集成 agent 的接入说明

当前选择与多模态等待主链以 [BRAIN_PROTOTYPE.md](BRAIN_PROTOTYPE.md) 及新包清单为准。早期 MiniCPM/OCR 指标只作为历史比较。

本目录提供 `PRTSCoreSession` 进程内入口与实际模型调用源码。Windows 已执行 Python、C++ 和 ONNX 对照；**没有在 Mac/Xcode/iPhone 编译，不能把本说明当作 Apple 验收通过**。当前阶段不需要重做 App 界面。

## 先运行可检查的构建

在 Mac 仓库根目录：

```sh
PRTS_CONTRACTS_ONLY=1 swift test --package-path apple/PRTSCore
bash scripts/build_apple_native.sh
```

第一条只编译 Foundation 契约、意图、等待和路线状态，不下载推理库。它比较 Python 冻结的 7 组标识、23 组命令、同车证据、叫号字段/广播及 627 个仿真位置；这是移植一致性检查，不是新感知精度。

第二条固定官方 llama.cpp 提交 `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`，构建 arm64 iPhone/Simulator XCFramework，再构建模型 Swift Package。生成的 framework 供 iOS 使用；Windows DLL 不能放进 App。脚本不签名或部署，也不会覆盖已有 XCFramework。

等待任务使用 `VisualWaitingBrain`，Vision OCR 只提供原始文字参考；旧 `QueueReader` 与数字拆分保留为对照。结构化生成通过新增 `prts_vlm_run_json`，需要从当前源码重建 framework。Vision OCR 与桌面 PP-OCR 不同，必须使用同一素材在 Apple 核对实际模型输出。

将 `apple/PRTSCore` 作为本地 Swift Package 加入现有工程，引入 `PRTSContracts` 与 `PRTSAppleModels`。项目最低 iOS 16；包里的 macOS 13 声明用于纯 Foundation 测试，并不表示 iOS 二进制能在 macOS App 运行。

## 模型位置和实例化

`CoreModelFiles` 接受八个本地 URL：分割 ONNX 及 manifest、检测 ONNX 及 manifest、语言 GGUF、视觉 GGUF、SenseVoice ONNX 及 tokens。按随包模型清单逐个核对 SHA256，再显式传入；不按文件名猜测量化版本。

当前桌面候选为 Qwen3-VL-4B-Instruct Q4_K_M 语言权重和 Q8_0 视觉投影器，使用官方 llama.cpp 固定提交、greedy 采样、关闭额外 CPU 权重重排副本。感知采用 Mask2Former Mapillary FP16 v2 与 YOLO11n ONNX；音频识别采用 SenseVoice INT8。具体文件与 SHA256 见 MODEL_MANIFEST.json。模型初始化放到后台，完成后再接传感器。旧 MiniCPM/OCR 架构的 20 分钟资源结果不能用于本版本，最新整套资源以 BUNDLE.json 为准。当前尚未证明包含前端的 8 GB 或 Apple 峰值。

```swift
import PRTSContracts
import PRTSAppleModels

let clock = { ProcessInfo.processInfo.systemUptime }
// files 由已校验的本地模型清单提供；webKey 由 App 自己配置。
let core = try PRTSCoreSession(
    files: files,
    maps: AMapService(key: webKey, city: "武汉"),
    useMetal: true,
    useCoreML: false, // 先建立 CPU 参考；CoreML 单独验证后开启
    clock: clock
) { event in
    // 快速转交前端事件队列。不要在此做同步渲染、文件上传或模型调用。
    eventSink(event)
}
```

没有地图服务时传 `maps: nil`，相机、问答和等待仍可使用；新路线查询不可用。高德 Key 不能写进本项目源码、日志、ZIP 或截图。临时 Key 不会替团队申请生产凭据。

## 前端提供什么

- `pushVideo(RGBFrame)`：转正且解除镜像的 RGB24、递增帧序号和会话时间。`CameraAdapter` 可转换转正 BGRA；Core 保留最新帧，不能每帧另建一个无限排队的推理任务。
- `pushAudio(PCMChunk)`：16 kHz、单声道 float32，推荐每块 100 ms，时间为首样本采集时刻。前端负责重采样、音频路由和实际回声消除。明确用户/PTT 标为 user，广播为 ambient；auto 不等于说话人身份识别。
- `pushText`：已完成 ASR 的文本或控件命令；环境广播必须指定 ambient。问答会保留已有导航和等待状态。
- `pushLocation`：明确 WGS84/GCJ02、精度、会话时间；相机真北朝向和精度可选。设备磁罗盘角不能未经胸前安装标定就当作相机朝向。
- `setPlayback`：反馈真正开始/停止播放的状态。没有回声消除时会抑制 auto/ambient 回授；user 输入仍可打断。

相机、PCM、定位与时钟要先映射到同一个单调会话时基。`close(completion:)` 异步等待正在执行的工作；停止传感器提交后调用，不阻塞主线程等待视觉编码结束。

## 前端消费什么

事件可由 `JSONEncoder` 输出与 Python 一样的扁平 envelope：`schema_version`、`sequence`、`emitted_s`、`type` 与载荷。`CoreEvent.speechRequest` 可直接交给 `AppleSpeechOutput`，无须 JSON 往返。

```swift
// 主线程创建与调用；speaker 由 App 保持强引用。
let speaker = AppleSpeechOutput(clock: clock)
speaker.playbackChanged = { active in core.setPlayback(active) }

// 在 eventSink 中切回主线程：
if let speech = event.speechRequest { speaker.accept(speech) }
if event.type == "speech_cancel" { speaker.cancelAll() }
if event.type == "sound_cue",
   let name = event.payload["cue"]?.string,
   let expires = event.payload["expires_s"]?.number {
    try speaker.playCue(name, expires: expires)
}
```

包里有自行生成的 `target_found`、`attention`、`arrived` WAV，前端可以替换。它们是事件提示，不表示通行或登车许可。音量和 AVAudioSession 路由由 App 统一管理。

`scene` 的 `semantic_grid` 提供 row-major uint8 类别图，base64 编码、宽高及原始类别表；可作为覆盖纹理显示。检测框和路径使用原始转正图的归一化坐标，OCR 框使用原图像素。所有覆盖层绑定 `frame_sequence`，不要直接画到下一帧。`guidance` 才是可发布引导；过时 `scene` 仍保留审计，但不作为当前路线。

## 当前与 Python 参考的明确差别

| 项目 | Apple 源码中的实现 | 必须回填的验证 |
|---|---|---|
| 感知 | 同一 ONNX + 纯 C++ 图像算子 | ORT iOS 实际版本、CoreML 分区和逐帧结果 |
| OCR | 系统 Vision 的准确模式，关闭语言纠错 | 它与 PP-OCRv6 不同，公交/文字所有用例重跑；不能引用 Python OCR 结果作为通过 |
| 局部路径 | 同一 C++ 连通搜索与完整折线校验，方向滞回 | 没有 Python 的光流路径注册，不能声称完整时序效果相同 |
| 等待 | 完整标识、同车绑定、OCR/VLM 一致性、候选与确认、去重/取消 | 新线路、车身号、方向、静态文字与 LED、噪声广播 |
| 路线 | 候选确认、步行步骤、坐标转换、终点距离、到达后等待 | 正确同一路段位置轨迹；偏离会报告重新规划，尚未自动重规划 |
| 音频 | 连续 PCM 分句、独立 ASR、系统 TTS 及提示音 | 真实麦克风回授、用户打断、前后台与音频路由 |

CoreML 默认为关闭，Metal 只控制原生视觉语言模型。电脑上 DirectML 的速度和内存不是 CoreML/ANE 的速度和内存。完整 App 需要记录进程 footprint、可用内存额度、热状态与持续延迟。

LiDAR、相机内参、位姿和深度仍是可选输入槽，本轮没有消费它们生成世界空间 AR 线。系统当前输出图像空间候选路径。`target_observed` 说明目标证据满足要求，并不证明车已停车或门已打开。

## agent 交接顺序

先执行 Foundation 对照并保留原始日志，再修复实际 Xcode 编译问题；接入现有采集和事件队列后，用随包相同输入做模型/任务回放。最后做手机端连续录制与内存测量，把具体 OS、设备、模型哈希、失败和改动写入 `APPLE_VALIDATION.md`。无法执行的步骤写“未执行”，不要改成成功，不以空回调或假数据填充模型结果。

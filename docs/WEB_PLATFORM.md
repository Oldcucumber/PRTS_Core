# PRTS Core Web 实时导航测试

更新：2026-09-24（本次测试跨 23／24 日）。入口：https://oldcucumber.github.io/PRTS_Core/

## 使用

页面默认不采集。点击「开始」后允许摄像头和麦克风，初始化本地模型并进行声音检查。首次打开会自动刷新一次以启用浏览器隔离；语音资源约 242 MB，所有静态资源合计约 443 MiB。未加载完成时会显示状态，不伪装成可用。

主画面仅显示实时摄像头、感知覆盖层、导航状态、最新提示和设备状态。点击「暂停」停止导航判断与播报，但保留麦克风以接收「继续导航」。点击「结束」释放视频、音频和推理 Worker。静音不停止感知。

本地口令：**暂停导航／停止导航、继续导航／开始导航、静音、恢复播报**。只接受完整短句，普通谈话中包含这些词不会直接执行。VAD 结束短句后由 SenseVoice 识别，连续采集不等于逐字流式识别。播报期间与结束后 300 ms 内的音频不送入识别；页面明确显示这一状态。此版不支持在播报中用语音打断，按钮始终可用。

镜头、推理后端、分辨率、深度辅助、覆盖层、前方监测角度、仅视觉调试、本地文件和日志导出位于二级设置。没有内置视频、路线演示、功能宣传卡或大脑调用。

## 运行链路

- 摄像头 → 最新帧采集 → 视觉 Worker（SegFormer-B0 + YOLO11n，同帧处理）→ NavigationSession → SpeechOutput → 设备本地中文 TTS。
- 麦克风 → AudioWorklet → 单声道 16 kHz PCM（100 ms 包）→ 独立 ASR Worker（Silero VAD + SenseVoice Small INT8 / sherpa-onnx v1.13.2）→ 限定口令 → 导航状态。
- 视觉忙时不积压待推理视频。音频一次提交一包，有界保留最多约 2 秒；解码积压时丢弃旧音频并重置 VAD，记录 audio_drop，不执行过期口令。
- 无连续通道稳定 1 秒后进入自由前进，只播报一次进入。前方障碍优先于路径方向；首次出现即提示，持续存在不重复，连续 1 秒消失后重新允许提醒。
- 普通方向稳定 500 ms 才提示，至少间隔 4 秒。障碍可打断方向提示；尚未播放的方向只保留最新一条。观测超过 2 秒不用于播报，连续失去新鲜结果会提示一次感知中断。
- 用户切镜头、结束或切入后台会取消旧会话及声音。没有本地中文音色时显示语音不可用；没有麦克风或隔离支持时明确显示语音输入不可用，可显式切换仅视觉调试。

Web 的模型与桌面 Mask2Former 不同，不宣称效果等价。地面通道、目标框和前方区域均为图像坐标；无米制测距、身体净空或世界空间 AR。前向默认半角 15°，用水平视场 60° 估算。大脑、地图和环境音语义识别不在本轮范围。

## 核心接口

页面由 bootstrap 初始化，app 负责视频适配；core-session、audio-input、speech-output 独立处理决策与声音，session-ui 只连接状态、按钮和适配器。

视频输入具有 observedAt（performance.now 毫秒）、ageMs 与即时感知结果；摄像头适配器用自身会话代数拒绝重启前返回的结果。PCM 包具有 observedAt、epoch、Float32Array samples，16000 Hz 单声道；ASR epoch 用于取消播报前或重置前的口令。

核心事件包含 type、session、sequence、emitted_ms。speech_request 包含 text、priority、replace_group、observed_ms、expires_ms；与 Python 核心的同名概念一致，但 Python 的时间单位是秒，此 Web 接口显式使用毫秒，不能原样交换。speech_cancel 清空待播与当前声音。播放端回传 playback_start / playback_end / playback_cancel / playback_error；导出的 trace 区分决策和真实播放回调。scene 仅保留最新观测，其他事件最多保留 1000 条。

当前优先级：前向障碍 100、感知中断 90、普通导航 40、启动声音 20。导航请求的有效期为来源观测后 2 秒；播放异常超过 10 秒取消并报告错误。

## 本地运行与回归

~~~powershell
npm ci
npm test
npm run build
npm run preview
# 首次使用浏览器测试
npx playwright install chromium
npm run test:static
npm run test:camera
npm run test:camera-selection
# Windows 中文本地音色 + ffmpeg；仅生成合成口令及既有录像的测试输入
npm run test:fixtures
npm run test:session
# 耐久测试
$env:SESSION_SECONDS='600'
$env:CAMERA_FIXTURE='outputs/asr/moving-camera.y4m'
node tests/session-browser.mjs
~~~

测试素材位于 outputs，不复制到 dist。可为 prepare-session-fixtures.mjs 传入自己的录像路径。源码保留历史录像，但发布页面不再提供它。

## 部署和验证边界

GitHub Pages 不提供自定义响应头；同源 isolation-sw 仅给响应附加 COOP/COEP/CORP，不缓存、不代理第三方内容。首次受控刷新后启用 SharedArrayBuffer。启动失败会保留视觉调试入口，不改用网络 ASR。构建包含所有模型和运行时，运行不向外部发送视频或音频；下载完成后的当前会话可断网继续，不承诺离线重新打开页面。

验证报告位于 docs/validation/20260923-realtime。自动测试将录像帧和本地合成语句通过浏览器采集 API 输入，运行真实视觉和语音模型、本地 TTS，并检查控制状态和轨道释放；不是现场盲人行走测试。播放回调证明浏览器已开始／结束播放，不证明扬声器在物理空间中的可听度。当前机器没有可用实体摄像头，iPhone/Safari 真机、室外噪声、不同说话人和现场障碍漏报率仍待验证。

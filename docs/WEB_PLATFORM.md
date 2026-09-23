# PRTS Core Web 小脑体验平台

更新：2026-09-23。发布目标为 main 分支的 GitHub Pages：https://oldcucumber.github.io/PRTS_Core/。

## 当前范围

| 功能 | Web 实现 | 边界 |
|---|---|---|
| 相机／视频输入 | 摄像头选择、室内短片、SANPO 街道片段、本地视频 | 相机需 HTTPS 与用户授权；帧不上传 |
| 地面分割 | SegFormer-B0 ADE20K，192×320 或 288×512 | 轻量模型，不是桌面 Mask2Former；floor / sidewalk / path 为候选，road 不计入 |
| 目标检测 | YOLO11n COCO，384×640，等比留边、类别内 NMS | 可标注车辆和交通灯类别，但不识别线路语义或灯色通行许可 |
| 固定实体 | 分割覆盖层与前向高置信度连通区域 | 低置信度或未识别区域不能当作安全区域 |
| 局部路径 | 近端连通区域中心线，检测框从候选区域中排除 | 图像坐标启发式，不提供人体净空、米制避障或世界空间 AR |
| 自由前进 | 无连续通道时进入，只提示一次；前方障碍出现提醒 | 半角可调 10–30°，默认 ±15°，假设水平视场 60° |
| 相对深度 | Depth Anything V2 Small，可选叠加或左右对比 | 非米制；不等同 LiDAR |
| 声音 | 系统 TTS、程序生成提示音、停止播放 | 默认关闭；紧急障碍提示优先 |
| 控制指令 | 文字开始／停止，可选浏览器语音识别 | 不是 SenseVoice；浏览器识别可能联网，不支持时用文字 |
| 路线 | 内置真实高德响应、确认目的地、进度回放、转弯／到达、JSON 导入 | 位置模拟，不对应示例视频；不提供真实定位跟随 |
| 在线地图 | 自带 Web 服务 Key 搜索 POI、选点查询步行路线 | 需网络、配额及浏览器可访问性；不内置临时 Key |
| 事件 | 实际导航／控制／路线事件，JSON 下载 | 最多 500 条；路线事件明确 simulated |
| 大脑 | Feature 卡片及未开放提示 | 无视觉语言模型、无问答／等公交／叫号识别，不伪造成功结果 |

## 模型与运行

ONNX Runtime Web 1.30.0 在 Worker 内执行。自动模式优先 WebGPU，启动失败回退 WASM；可强制指定。Pages 没有跨源隔离头，WASM 使用单线程。分割、YOLO 与可选深度处理同一采集帧，保持完整视野并显示真实端到端耗时；超过 2 秒的结果仅展示，不播报即时导航。

分割输出新增 semantic_confidence，既有 floor_probability、winning_class 和学习权重保留。scripts/extend_web_semantics.py 可为旧导出追加此输出；export_web.py 新导出包含此输出。YOLO 导出与桌面检测权重一致，但浏览器路径算法不是桌面算法逐项移植。网络缓存和设备算力影响首次启动及帧率。

模型、WASM、样例和脚本从同站点加载；可选浏览器语音识别、在线高德查询是明确的联网功能。首次加载仍需下载资源，不声称安装前完全离线可用。

## 路线导入

坐标必须为 GCJ02；每个步骤有非空 points，总计至少两个不同的点。浏览器仅绘制无底图路线示意。

```json
{"destination":{"name":"示例目的地"},"steps":[{"instruction":"沿路线前进","points":[{"longitude":114.39,"latitude":30.43,"crs":"GCJ02"},{"longitude":114.391,"latitude":30.431,"crs":"GCJ02"}]}]}
```

## 开发与部署

```powershell
npm ci
npm test
npm run build
npm run preview
# 完整浏览器回归首次需要安装浏览器
npx playwright install chromium
npm run test:static
```

build 清理并重建本仓库 dist，输出独立静态站点。main 推送触发现有 Pages 工作流，先运行单元测试，再构建和部署。仓库子目录访问纳入回归测试；截图和运行中间数据放在 outputs。

## 验证与限制

20 项 JavaScript 单元测试覆盖原地面／深度逻辑，以及前方扇区、单次进入播报、障碍重现、检测 NMS、语义障碍与路径排除。已在真实 Chromium 中实际运行 WebGPU、WASM 和深度对比，验证文字停止、未开放大脑请求、路线确认与到达、事件导出和 390px 手机视口无水平溢出。手机视口测试使用桌面浏览器，不是 iPhone 真机性能。

语音接口可用性和 TTS 音色因浏览器系统而异。真实摄像头画质、弱光、小障碍、手机持续发热仍需现场评估。高德查询提供接口与错误反馈，未使用访客 Key 进行实地路线验收。来源和许可见 THIRD_PARTY_NOTICES.md。

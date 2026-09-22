# PRTS Core 流式接口 v1

当前大脑架构见 [多模态原型](BRAIN_PROTOTYPE.md)。默认等待链路由视觉语言模型解释事件，旧 OCR 字段匹配仅用于历史对照。

状态：Python 参考实现已实际运行；Apple 的 `PRTSCoreSession` 运行入口与模型、任务、地图调用源码已写入，其编译及设备验证未执行。核心是进程内组件，不会自动打开摄像头、麦克风、扬声器或网络服务端。

## 输入、时钟与线程

所有 `timestamp_s`、`observation_s`、`emitted_s`、`expires_s` 都是同一会话单调时钟中的秒。Python 默认 `time.monotonic()`；前端必须把相机、音频、定位各自的时间映射到该时钟，不直接混入 Unix 时间或视频文件相对时间。PCM 时间表示块首样本的采集时刻，整块采集完成后才送入。

| 输入 | Python API | 必需语义 |
|---|---|---|
| 相机 | `push_video(VideoFrame)` | 严格递增帧序号、时间戳；已转正 BGR uint8 H×W×3；提交后不得修改缓冲 |
| 连续音频 | `push_audio(AudioChunk)` | 16 kHz float32 单声道、每块最多 1 秒；通常 100 ms；块序号连续 |
| 明确文本命令 | `push_text(text, role)` | 默认 user；用于文字入口和前端已经完成的 ASR，不要求重复识别 |
| 位置与朝向 | `push_location(Location)` | 最新位置、WGS84/GCJ02、水平精度米；相机真北朝向度与精度可选 |
| 已归一化位置 | `update_location(Location)` | 同步低层入口；调用方已经完成坐标转换，适合确定性回放 |
| 实际播放状态 | `set_playback(bool)` | 前端真正播放/结束时回传，不能把“已请求 TTS”当作正在播放 |

图像采用最新帧槽。ASR 和语言请求各保留最多 4 个待处理任务，满时丢弃旧任务并发出 `input_dropped`。语音分句使用能量 VAD，保留短预卷、约 450 ms 静音结束和最长 12 秒上限；它不是说话人识别。摄像头、PCM 生产者分别串行提交；事件回调应快速入队，不要在回调内部重新调用核心或阻塞推理线程。

`role=user` 是前端明确的用户/PTT 通道，能打断播报；`ambient` 只作为等待任务的广播证据；`auto` 使用可识别命令语义区分。无回声消除且正在播放时，auto/ambient 音频被抑制。`echo_cancelled=true` 仅表示前端已处理回声，不能据此认为已辨认说话人。连续环境音中的用户/他人区分仍需实际设备验证。

`push_location` 将商业服务的坐标转换放到独立工作线程，期间不会阻塞相机输入；高德在边界把 WGS84 转 GCJ02 一次，GCJ02 输入直接使用。转换失败时发出 `map_unavailable`。目前没有获准离线路网，因此全新目的地检索、规划和需要服务的坐标转换仍要求网络；模型、OCR、ASR 和已有等待任务不依赖网络。

## 输出与几何

每个事件含 `schema_version=1`、会话唯一递增 `sequence`、`emitted_s` 和 `type`。`emitted_s` 是结果可用时间；`observation_s` 是证据采集时间，二者不可互换。

| 事件 | 内容与前端用途 |
|---|---|
| `scene` | 证据帧序号、区域、检测、局部路径、处理耗时及证据年龄；用于覆盖层和记录 |
| `guidance` | 当前可发布的方向/路径/文本；过时场景可能被降为 WAIT，不得直接播报 scene 中的旧路径 |
| `destination_candidates` | 候选地点、地址、序号和确认要求；只有确认后才规划目的地路线 |
| `route_started` / `route_progress` | 路线点串、坐标系、版本、步骤、转向、定位精度及到达状态 |
| `wait_started` / `wait_cancelled` / `wait_continued` | 等待开始、取消与明确提醒尾句的延续；延续不会重置开始时间或已提醒状态 |
| `target_evidence` | 当前画面的原始 OCR 工具输出与证据时间；不直接表示已找到 |
| `brain_observation` | 多模态观察、实际标识、事件含义、决定、任务版本、证据年龄和原始模型输出；用于解释等待判断 |
| `target_candidate` | 历史规则链路的候选事件；默认多模态链路通过 brain_observation 表达不确定 |
| `target_observed` | 满足目标证据条件，去重提醒；不表示车辆已停车、车门已打开或可以登车 |
| `transcript` / `answer` | 实际识别文本或视觉答复、输入/证据时间；answer 携带 retrospective 标志 |
| `speech_request` | 文本、优先级、过期时间、replace_group，供离线系统 TTS 消费 |
| `speech_cancel` | 取消此前未结束的播报；前端立即停止声音，不等待模型算完 |
| `sound_cue` | cue 名称、关联 speech sequence 和过期时间 |
| `worker_error` / `request_error` | 哪条运行链路出错；记录并展示开发状态，不能用伪造结果填补 |

原始检测框 `[x1,y1,x2,y2]`、区域轮廓和候选路径均使用**已转正输入图像的归一化坐标**，左上角 (0,0)，右下角 (1,1)。OCR box 是同一输入的像素坐标。前端 letterbox/裁切后必须应用对应显示变换。保持证据帧与覆盖层配对，不把旧帧路径直接画到另一个时间的画面上。

历史 legacy 模式的叫号 OCR 可携带 `queue_verification_attempted`、`queue_role`、`queue_heading` 和标题四角。角色为 called / waiting / counter / queue_label / unconfirmed / other；只有满足整套任务条件后发出的 target_observed 才代表确认。可见间距拆分字段标记 `number_source=ocr_pixel_gap_split` 并保留 `original_text`，供比赛演示追溯数字来源。

区域分为可走、道路/特殊通行、固定实体、动态物体、未知，并保留原类别。细小区域/轮廓可能为显示而简化；局部路径依据稠密网格，不能拿显示轮廓重新计算通行。`CANDIDATE` 是图像空间候选，`WAIT`/`STOP` 不输出可走承诺。没有相机空间注册时，不能把归一化路径换成虚构的米制路线。

地图朝向是相对真北的水平角，图像左右是相机视角。只有提供质量足够的相机朝向时才输出 `relative_bearing_deg`。手机朝向、道路走向与人的移动方向不是同一个量。

## 自由前进模式

没有连续人行道且不是传感器过时、镜头运动或地图冲突时，`guidance.mode=free_forward`。前方未检测到障碍时 `status=FREE`，检测到障碍时为 `STOP`。`path` 保持空，不构造已经验证的通道。`scene.guidance` 每帧持续更新，前端不依赖语音刷新覆盖层。

`forward_scan.sector_box` 是归一化图像监测区域；`half_angle_deg` 默认 15，`horizontal_fov_deg` 默认 60，`angle_source=assumed_horizontal_fov`。角度可通过 Guide/LocalGuide 配置，当前没有相机标定或米制距离。检测框与高置信度的固定/动态语义区域共同检测前方障碍，忽略监测区外及高处区域；“未检测到”不被输出成已验证无障碍路径。

每次进入自由前进只提示一次；持续空旷、持续同一障碍均不重复播报。障碍由无到有时提示，障碍消失后再次出现会重新提示。找到连续人行道后返回原导航；仅结果过时不算退出自由模式，不重复宣布进入。Apple 源码同步该契约，尚未真机编译。

## 播报和证据寿命

speech_request 的优先级数值越大越优先；相同 `replace_group` 替换旧请求。导航组的更新可以中断旧导航句子，明确用户输入取消此前播报。客户端在声音开始前及播放过程中检查 expires；记录实际开始/结束/取消时间供延迟评估。

参考核心对行进指导使用 2 秒新鲜度；当前多模态公交证据上限 30 秒，叫号上限 45 秒。brain_observation 的 evidence_s 是实际帧或环境转写的时间，observation_channel 区分 vision 和 environment_transcript。慢 CPU 结果只能作为带年龄的回顾性提醒，不能表示目标此刻仍在。DirectML 是已测试的连续导航路径，CPU 地面模型较慢时指导会为 WAIT。延迟较长的问答会在播报中说明“根据约几秒前的画面”。这些阈值是当前实现策略，不等于行业安全标准。

新出现的 STOP/WAIT 播报优先级 80，普通方向变化 40，已播过的相同导航状态重复提醒 20；目标提醒 70，普通答复 30。同一状态的重复提示不再打断长答复，新的危险仍会打断。目标提醒只读目标与事件类型，完整模型观察另存 evidence_text，避免把单张照片中的模型运动猜测读成事实。

默认 TTS 路径：核心输出文字，Apple 前端用 AVSpeechSynthesizer；Windows 验证器在运行中合成 WAV 并记录可用时间。模型原始音频输出尚未接入，不发送虚假的 audio_chunk。提示音由前端资源映射播放，不从自由文本生成任意文件路径。

## 可选空间数据

SpatialSample 与帧使用同一时钟并带 `world_frame_id`。内参为转正输入像素对应的 row-major 3×3；相机到世界是 row-major 4×4，米制右手系，相机采用 ARKit 约定的 +x 右、+y 上、观察方向 -z。图像 +y 向下，投影时必须显式翻转，不能直接把原 OpenCV +z 前方位姿塞入。

深度为米，明确分辨率、与彩色图的配准和置信度；无效样本为 NaN。重定位或世界原点重置时换 world_frame_id。Python mesh 字段当前是预留载荷，尚未消费；Swift 尚未定义最终网格缓存协议。本阶段缺少这些字段仍可启动，但不产生经过验证的世界空间 AR 引导。

## 直接运行

Apple 的输入为 RGB24，Python 为 BGR ndarray；Apple ASR/语言队列各保留一个最新待处理请求，Python 各保留四个。Swift `scene.semantic_grid` 是 base64 uint8 类别纹理，Python `regions` 是显示轮廓，两者都绑定原始证据帧。Swift 使用 Apple Vision OCR，局部引导暂未移植 Python 光流注册；相同事件名称不等于两个平台的感知精度已经一致。完整调用方式与已知差异见 [Apple 接入说明](APPLE_INTEGRATION.md)。

参考入口是 `scripts/replay_stream.py`，场景 JSON 记录素材来源、源时间、裁切、遮挡旧 UI 和音频块时间。模型组合必须显式指定，避免无意使用第一版默认权重。实测资源、模型清单与 Apple 剩余工作分别见实施报告、模型报告及 `apple/README.md`。

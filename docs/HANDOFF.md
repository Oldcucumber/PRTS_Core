# PRTS Core 本轮交接

> 2026-09-23：Web 已改为摄像头主界面的实时小脑测试，接入本地 VAD／SenseVoice 与导航播报，移除展示视频和路线面板。接口、回归与现场未验收项见 [WEB_PLATFORM](WEB_PLATFORM.md)。大脑仍不在 Web 运行；下文桌面核心的延迟和 Apple 限制仍有效。

更新：2026-09-22。此文是下次继续工作的起点。本轮收尾仅整理文档、保存源码与验证证据，不再新增模型或展开加速实验。

## 已完成

- Python 后端分为持续地面/障碍感知的小脑，以及处理即时问答、持续等待的大脑；二者通过任务版本、帧时间和事件调度协作。
- 连续人行道缺失时进入 `free_forward`，只在进入时提示；默认监测前方约 ±15°（假设水平视场 60°），前向障碍出现时提醒，持续同一状态不重复。恢复人行道后返回路径导航。
- 真实连续公交视频完成：等 20 路提醒成功；相同视频等 20A 没有误提醒。没有只按 OCR 字符串触发等待完成。
- MiniCPM-V 4.6 Q5+Q8 与 Qwen3-VL-4B Q4+Q8 的同 22 用例比较及原始结果已保留。当前暂留 Qwen，以实际等待判断效果为依据。
- Python 63 项测试通过；24 个 Swift 文件通过语法检查，但这不等于类型检查、编译或真机运行。
- v2 ZIP 独立解压后，908 个文件散列校验、63 项测试、2 个实际多模态案例、原始告示录音/视频回放均通过。已有 Windows Python 依赖被复用，不是全新操作系统验收。

## 尚未解决

1. 大脑 CPU 判断约 29 秒。原始 912 录像先观察到路面，推理结束时片段已结束，未观察后续车辆，因此漏提醒。综合回放的 101 虽识别正确，32.375 秒证据年龄超过 30 秒车辆限制，未提醒；C002/B003 提醒成功。三目标综合验收仍是失败。
2. 桌面保守核心峰值 7.708 GB，加假设 2 GB 前端为 9.708 GB；不是 Apple 内存实测，整套 8 GB 尚未达标。
3. 图像路径、地面分割仍有误差；没有米制人体净空、真实空间 AR 注册或可依赖的动态避障闭环。LiDAR、深度、AR 位姿保留接入契约。
4. Apple 编译和真机验证尚未进行；没有原生音频生成模型，当前文字交给 TTS。真实地铁场景尚未验证。
5. 高德路线查询依赖网络和有效 Key；已有演示是真实路线响应配合模拟 GPS，并非真实街景定位导航。临时 Key 不提交，续作时不能假定仍有效。

## 下次优先级

先读 [延迟分析](LATENCY_ANALYSIS.md)，完成大脑硬件加速与分阶段性能验证；再用真实 912 和 20/20A 连续录像验证新帧调度，重新测量内存。保留模型判断负责等待语义、小脑负责基本导航的职责。MiniCPM 仍是重点候选，当前同协议 CPU 对比不代表端侧优化后的最终排序。

继续遵守：效果先于帧率；不过度防御性编程；保持前端无关；不得将照片拼接、模拟位置、旧观察或脚本音频当作实地实时效果。无需另造演示来掩盖现有失败。

## 关键代码

- `prts_core/guidance.py`、`forward_obstacles.py`：路径与自由前进。
- `prts_core/streaming.py`：并行流式调度、证据年龄与播报。
- `prts_core/waiting_brain.py`：v9 多模态等待协议；OCR 仅作辅助证据。
- `prts_core/native_vlm.py`、`native/prts_vlm.cpp`：本地模型 C ABI。
- `apple/PRTSCore/`：Swift 源码与集成契约。
- `scripts/build_brain_bundle.py`：默认保留三目标验收门槛；显式 `--include-known-demo-failures` 只允许携带失败结果打包，BUNDLE 必须标记 failed acceptance，不会把失败改成通过。

## 本机交付与复现

后续直接在 `D:/gpd/PRTS_Core` 的 `main` 分支继续，源码与文档已合并并推送到远端 `main`，临时 `codex/prts-core-prototype` 分支已删除。原实验工作树 `C:/Users/JIONG/.codex/worktrees/3a3e/PRTS_Core` 保留为脱离分支的历史快照，其中的虚拟环境、模型和运行日志仍然保留；这些忽略提交的资源不会因 Git 合并自动移动到主目录。

交付目录：`D:/gpd/PRTS_Core_Delivery/20260922_revision/`，包括四段演示、原理图、修订报告及 `PRTS_Core_brain_20260922_v2.zip`。ZIP 4,956,788,307 字节，SHA256：`ef36cf01253a8f63f06b23c70f8a4ac84a685978e84f11a96de3167aca8bbfb8`。这次 Git 收尾新增的索引、交接及延迟文档不回写冻结 ZIP，核心运行代码与该包一致。

独立解压目录：`D:/gpd/PRTS_Core_Validation/brain-v2/`。完整媒体、模型和运行日志保留在交付包与原工作树，不上传 Git。只克隆仓库时先跑逻辑测试；完整回放需要按 [包说明](BRAIN_BUNDLE_README.md) 准备本地资源。

本次运行配置：官方 llama.cpp 提交 `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`，`libprts_vlm_lowmem.dll`，6 CPU 线程、greedy、4096 上下文；Qwen Q4_K_M 语言 + Q8_0 视觉；Mask2Former FP16 与 YOLO11n ONNX 走 DirectML；SenseVoice INT8；OCR profile `bounded_upright_no_arena`。

完整 Python 核心回归命令：`.venv-dml/Scripts/python.exe -m unittest discover -s tests -q`。此前的结果表及旧脚本只对应各自阶段，先阅读文档索引再复用。

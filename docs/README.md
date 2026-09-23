# 文档索引

以本页列出的当前交接与接口为入口。历史实验中的默认模型、资源数字和未完成事项只对应当时的版本。

## 当前交接

- [WEB_PLATFORM](WEB_PLATFORM.md)：2026-09-23 Web 小脑体验功能、运行和验证。

- [HANDOFF](HANDOFF.md)：下一次从哪里继续。
- [REVISION_FREE_FORWARD](REVISION_FREE_FORWARD.md)：自由前进、真实公交演示和 MiniCPM/Qwen 对比。
- [LATENCY_ANALYSIS](LATENCY_ANALYSIS.md)：29 秒推理延迟的实际组成。
- [BRAIN_BUNDLE_README](BRAIN_BUNDLE_README.md)：含模型 ZIP 的复现方法。
- [validation/20260922](validation/20260922/README.md)：随仓库保留的验证证据。

## 接口和实现

- [INTERFACE_SPEC](INTERFACE_SPEC.md)：帧、PCM、位置、事件、取消、TTS 和自由前进契约。
- [BRAIN_PROTOTYPE](BRAIN_PROTOTYPE.md)：大脑与小脑协作实现。
- [APPLE_INTEGRATION](APPLE_INTEGRATION.md)、[NATIVE_PORTABILITY](NATIVE_PORTABILITY.md)：Apple/C ABI 接入及尚未验证部分。
- [原理图](diagrams/05-brain-cerebellum.svg)：当前讲解图；旧编号图保留早期方案。
- [MEDIA_CREDITS](MEDIA_CREDITS.md)、[STAGE2_THIRD_PARTY_NOTICES](STAGE2_THIRD_PARTY_NOTICES.md)：数据来源与上游许可。

## 历史实验记录

FINAL_VALIDATION 描述第一版大脑包，以自由前进修订报告补充其后续结果。PRTS_CORE_RESULTS、BUS_GENERALIZATION、QUEUE_VALIDATION、RUNTIME_RESULTS 和 INTEGRATION_BUNDLE_README 包含旧模型或 OCR 规则阶段结果，不能替代当前多模态等待方案验证。PRTS_CORE_PLAN、PRTS_CORE_STAGE2_PLAN、RAPID_PROTOTYPE_PLAN、PRTS_CORE_STAGE2_PROGRESS 保留规划与探索历史。

`scripts/prepare_*` / `download_*` 为资源准备；`probe_*` / `evaluate_*` / `compare_*` 为研究；`replay_stream.py` 为流式回放；`compose_stream_demo.py` 为按实际时刻合成展示；`build_brain_bundle.py`、`run_bundle_replay.py`、`validate_extracted_brain.py` 为打包与独立复现。

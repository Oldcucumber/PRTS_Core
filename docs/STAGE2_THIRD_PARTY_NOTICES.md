# 第二阶段组件与演示来源

仓库自身沿用根目录 `LICENSE`（GPL v3），没有把模型和数据改成仓库许可证。完整文本及获取来源保存在 `licenses/stage2/`；文件与权重哈希随集成包提供。

| 组件 | 来源与适用文本 |
|---|---|
| MiniCPM-V 4.6 GGUF | [官方模型卡](https://huggingface.co/openbmb/MiniCPM-V-4.6-gguf)，Apache 2.0；固定 revision 与量化衍生过程另有清单 |
| Qwen3.5 2B / 4B 对照 | [官方模型](https://huggingface.co/Qwen/Qwen3.5-4B) 与 [Unsloth GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF)，Apache 2.0；固定权重 revision 和 SHA256 随模型 manifest |
| Qwen3-VL 4B Instruct 当前候选，2B 对照 | [官方 GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF)，Apache 2.0；语言 Q4_K_M，视觉 F16/Q8 比较，模型卡与来源记录在 licenses/stage2 |
| Mask2Former Mapillary 权重 | [官方模型仓库](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md) 对权重使用 CC BY-NC 4.0；代码为 MIT，不能以代码许可替代权重许可 |
| YOLO11n 及其 ONNX 导出 | [Ultralytics](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)，AGPL v3；原权重、导出源及修改随包记录 |
| llama.cpp-omni | 固定提交的原生推理源码，MIT；随包保留 PRTS C ABI 修改源码与上游获取脚本 |
| 官方 llama.cpp 当前运行时 | [上游](https://github.com/ggml-org/llama.cpp)，MIT，固定提交 b29c606e28a01b1bc8c1351026a0fa6e616bf6c4；新增 PRTS 结构化生成与确定性采样接口 |
| ONNX Runtime | MIT；桌面通过的版本与 Apple 包二进制版本分开记录 |
| sherpa-onnx | Apache 2.0 运行库 |
| SenseVoice 权重 | FunASR Model Open Source License Agreement v1.1，见随包原文；不能因运行库为 Apache 2.0 就把权重也标成 Apache |
| RapidOCR / PaddleOCR | Apache 2.0；桌面使用 PP-OCRv6 small，Apple 源码使用系统 Vision，二者不是同一模型 |
| 提示音 | 本项目脚本生成的正弦波资源，随生成脚本及哈希提供 |

演示数据分别保留来源。SANPO 是 CC BY 4.0 的公开研究数据；不同视角、人工标签与传播标签在报告中区分。Wikimedia Commons 的各文件有独立作者及 CC BY-SA 2.0/3.0/4.0 等条款，随 `media-sources/` 的原始 metadata 保存，衍生标注图或视频仍遵循相应条款。作者署名不因裁切或添加覆盖层而移除。

毕设四段视频是用户提供的私人素材；包中使用的片段与录音仅作为本团队开发、验证和演示材料，未上传或发布。真实高德路线来自用户授权的服务请求，演示中跟随路线的位置是明确标注的仿真输入，不是现场 GPS 或街景对应。模型包不包含高德 Key、地图 SDK 或可转售的商业街景数据。

分发前保留这些原文及各媒体署名；当前包面向研究与团队集成。权重的非商业条款也适用于其量化/导出版本。

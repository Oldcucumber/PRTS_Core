# 便携核心的实现与验证

此页区分“已实际执行的 C ABI”和“待 Mac 构建的 Swift 源码”。整体目标仍在实施中。

| 模块 | 实现 | 已验证范围 |
|---|---|---|
| 视觉语言 | llama.cpp-omni，PRTS C ABI | Windows CPU 真图推理、UTF-8 分片、取消及再次调用 |
| 路径几何 | 纯 C++，无 OpenCV/框架依赖 | 39 张构造掩码及 14 张实际模型掩码，各 3 个方向偏好，共 159 组路径与 Python 逐像素完全一致；另有 4 项完整折线校验 |
| ONNX 推理 | C API，注入调用方唯一 ORT 实例 | YOLO 与 Mask2Former FP16 真实导出模型、同一实际图片张量；原生与 Python 输出完全一致 |
| RGB 和概率网格 | 纯 C++ 双线性缩放、letterbox、概率裁切/归一化 | 8 个尺寸组合数值检查通过；另在 14 个既有感知输入上通过预设模型迁移门槛 |
| 叫号数字间距 | 纯 C++ 透视取样、Otsu 和字形间距 | Windows x64 实编译，11 组固定输入与 Python 分组一致；不识别新数字，不证明 Apple OCR 效果 |
| Swift 模型组合和任务调度 | `PRTSCoreSession` 串起实际感知、ASR、VLM、Vision OCR、任务与地图 | 已有源码；语法树检查通过，**不等于 Swift 编译、API 类型检查或 Apple 运行通过** |

掩码比较验证的是代码移植一致性，不是地面语义的准确率。39 张构造图不计为真实场景；14 张模型掩码也不自动成为独立人工真值。

证据分别位于 `outputs/stage2/native-geometry/actual-masks-v1/`、`native-onnx/semantic-v1/`、`native-image/numeric-v1/` 和 `native-image/model-v1/`。图像算子在同一 DirectML FP16 模型的 14 帧上：平均类别一致率 99.4182%、最低可走掩码 IoU 99.4915%、人工可走 IoU 最大下降 0.03158 个百分点，达到原定 98% / 95% / 1 个百分点门槛。它仍是既有开发/导出样本的迁移检查。报告记录源代码和 DLL 哈希。

## 运行库边界

`prts_onnx_create` 接收应用已有的 `OrtGetApiBase()` 指针，并请求 API 23。桥接不链接或下载另一份 ORT。官方 [C API](https://onnxruntime.ai/docs/api/c/struct_ort_api.html) 的环境、会话和张量均在同一运行库创建及释放。

`sherpa-onnx` 1.13.8 依赖 ORT Swift 包 1.28.2；[该包清单](https://github.com/csukuangfj/onnxruntime-libs/blob/v1.28.2/Package.swift) 固定的 iOS 归档是 1.28.1，SHA256 `992d8a0cc6014cccc3a7815c36bbff5e5a06833ea2c4d47dd43ef071f639cf9d`。已下载检查 C 模块名 `onnxruntime`，但未执行 Apple 二进制。

CoreML provider 是显式选项，初始参考采用 CPU；按照 [官方 CoreML 配置](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html) 使用 MLProgram。代码中可选择 CoreML 不代表整个模型都运行于 ANE，也不代表手机的内存或速度已经验证。

## 本机复现

使用 Zig 或兼容 C++17 编译器分别构建 `native/portable/` 的源文件；包含 `portable/include`，ONNX 桥接额外包含 `native/third_party/onnxruntime`。然后运行：

```powershell
.venv-core-cpu/Scripts/python.exe scripts/validate_native_geometry.py --mask-dir outputs/stage2/perception/cpu-validation/dml-fp16-v2 --output outputs/new-geometry-check
.venv-core-cpu/Scripts/python.exe scripts/validate_native_onnx.py --output outputs/new-onnx-check
.venv-core-cpu/Scripts/python.exe scripts/validate_native_image.py --output outputs/new-image-check
```

Mac 构建工具会将便携模块编入 `prts_vlm.framework` 并公开 C 头。Swift 调用方读取模型 manifest、接收转正 RGB，输出语义网格和检测框；路径模块接收当前可走掩码。新增 `PRTSCoreSession` 使用分开的相机、ASR、语言、事件队列，接收连续 PCM、位置和控制输入；已写入目的地确认、路线进度、到达后等待、同车证据、语音打断和过期处理。未使用空推理回调替代模型。

移动源码的 OCR 使用 Apple Vision，局部路径只包含当前掩码搜索及方向滞回，尚无 Python 光流注册。它们仍要在目标系统上做效果对照。627 个路线位置等 Python 契约结果已冻结为 Swift 测试资源；测试代码还没有在 Swift 执行。具体接入见 [Apple 集成说明](APPLE_INTEGRATION.md)。

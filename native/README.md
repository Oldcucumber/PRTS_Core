# PRTS 原生视觉语言接口

## 当前官方运行时

多模态大脑原型使用官方 `https://github.com/ggml-org/llama.cpp` 提交 `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`。配置时传 `-DPRTS_OFFICIAL_RUNTIME=ON`。当前交付 Windows 产物是 `libprts_vlm_lowmem.dll`（`PRTS_LIBRARY_NAME=prts_vlm_lowmem`、`PRTS_CPU_REPACK=OFF`，此前对照产物为 `libprts_vlm_official.dll`）；Apple 脚本同样固定该提交，但 Apple 编译与运行尚未执行。

`prts_vlm_run_json` 接收 JSON Schema 并使用运行时语法约束。`prts_vlm_set_deterministic(handle, 1)` 在请求间选择贪婪解码、无 presence penalty；不调用则采用温度 0.7/top-p 0.8/top-k 20/presence 1.5。采样设置必须跟随实测模型配置，不把某一模型的推荐设置自动套到所有模型。实际 model/projector/运行时 SHA256 在交付包中记录。

下面 MiniCPM 专用 fork 的记录属于上一阶段对照，可通过关闭 `PRTS_OFFICIAL_RUNTIME` 重建，不是当前默认 Apple 构建来源。

`prts_vlm.h` 是 PRTS 自己的 C ABI，供 Python 研究端和 Apple Swift 桥接调用同一实现。它接收 UTF-8 提示词及可选 RGB24 图像，逐字节片段回调输出，支持取消，每次请求清空上下文以限制内存和陈旧状态。不会打开网络、摄像头或扬声器。

本机已用 Zig 0.16.0 / CMake 4.4.3 / Ninja 1.13.2 在 Windows 编译 `libprts_vlm.dll`，并用真实公交画面完成 CPU 推理。未在 Mac/Xcode/iPhone 构建或实测。

## 固定依赖

- 源码：`https://github.com/tc-mb/llama.cpp-omni`，提交 `13401aa8480d68a725e131cc4dbe5eda1f996a7f`，MIT。
- 这是 OpenBMB MiniCPM-V-Apps 提交 `d1aac05eecaac64b7fc35f1cc93a06b726ce10e3` 指定的推理子模块。
- 权重由 `scripts/prepare_gguf.py` 获取：当前默认语言 Q5_K_M + 视觉 FP16，Q4 与 F16 保留作对照，固定源 revision 与校验值。不可随意替换不同版本 projector。
- CMake 只构建所需的 llama、common、mtmd 和 PRTS 桥接；不依赖这个 fork 的 HTTP server 或 Omni 音频组件。探索构建 server 时发现的 vendor 问题不影响本接口，最终包不包含未经验证的 server。

## 构建

```sh
git clone https://github.com/tc-mb/llama.cpp-omni Vendor/llama.cpp-omni
git -C Vendor/llama.cpp-omni checkout 13401aa8480d68a725e131cc4dbe5eda1f996a7f
cmake -S native -B build/prts-native -DLLAMA_SOURCE_DIR="$PWD/Vendor/llama.cpp-omni" -DCMAKE_BUILD_TYPE=Release
cmake --build build/prts-native --target prts_vlm --parallel 4
```

macOS 默认启用 Metal，CPU 可传 `-DGGML_METAL=OFF`。iOS 需要使用 iPhoneOS/iPhoneSimulator SDK、arm64 目标和相应最低版本分别构建，再制成 XCFramework；随包 Apple 构建工具负责这一层。不能将 Windows DLL 放进 iOS 应用。

## 调用约定

1. 在后台工作队列调用 `prts_vlm_create`；路径为本地 UTF-8，首次模型准备完成后可离线。
2. `prts_vlm_run` 同步占用调用线程，逐片段回调。每个 handle 一次只运行一个请求；回调不得重入生成接口。
3. 回调是 UTF-8 **字节**片段，可能拆开一个汉字；接收端用增量解码器。Python 绑定已处理。
4. `prts_vlm_cancel` 可从其他线程调用；语言解码可中断，正在进行的视觉编码会在返回后取消，不能宣称零延迟停止计算。交互层可先停止播放和丢弃旧 generation 的输出。
5. 停止提交新请求后调用 `prts_vlm_destroy`。图像借用期间调用方不得修改或释放内存。

返回值 `0` 为生成结束、`1` 为取消、`2` 为达到输出长度上限、`-1` 为错误。非 `0` 的输出不能自动视为完整结构化答案。

原生接口当前负责视觉语言推理；完整感知、地图和任务调度仍由 PRTS Core 上层组合，不应把此单库描述为整个产品。

现新增 `portable/`：路径几何、ONNX 调用与图像变换 C ABI。构建时一并导出供 Swift 调用；各模块的 Windows 实测和 Apple 未验证范围见 [便携实现记录](../docs/NATIVE_PORTABILITY.md)。旧 Windows 视觉语言 DLL 不含这些新增符号，数值验证目前使用分别构建的 `prts_geometry.dll` / `prts_onnx.dll` / `prts_image.dll`，不可混淆。

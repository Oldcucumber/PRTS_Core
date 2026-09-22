# Apple 核心接入源码

入口为 `PRTSCoreSession`：相机与 PCM 输入，感知、问答、持续等待、目的地确认和路线状态，输出版本化事件、TTS 请求及提示音。没有创建前端页面。**这些 Swift 源码尚未在 Mac/Xcode/iPhone 编译或执行，当前交付状态是待团队构建验证。**

在团队 Mac 的包根目录运行：

```sh
PRTS_CONTRACTS_ONLY=1 swift test --package-path apple/PRTSCore
bash scripts/build_apple_native.sh
```

先比较纯 Foundation 状态机，再构建原生 iOS framework 与完整 Swift Package。Windows DLL 只供桌面参考，不能复制进 iOS。构建脚本固定上游版本，不签名或部署。

模型、输入时钟、线程、调用示例、TTS 和前端交接步骤见 [Apple 接入说明](../docs/APPLE_INTEGRATION.md)。对外数据语义见 [接口契约](../docs/INTERFACE_SPEC.md)，原生数值证据见 [便携实现记录](../docs/NATIVE_PORTABILITY.md)。

源码里的 Apple Vision OCR 与 Python 的 PP-OCRv6 不同；Swift 局部路径尚无 Python 光流注册，所有公交和连续导航用例必须在 Mac/设备重跑。LiDAR、位姿和深度保留接口，尚未生成经实测的世界空间 AR 路径。CoreML 默认关闭；桌面 DirectML 结果不能外推为 ANE 性能。完整应用的内存、热状态、音频回授和后台行为需由真机补证。

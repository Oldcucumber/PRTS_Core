# 浏览器地面识别与方向测试台

## 启动

要求 Node.js 20 或更新版本。项目已包含两个导出的 ONNX 模型，每个约 15.2 MB。

```powershell
npm install
npm start
```

打开 http://localhost:8080 后默认请求后置摄像头并显示预览，点击底部“开始”启动推理，运行时该按钮变为“停止”。顶部半透明面板显示 FPS、推理耗时、总耗时、帧数、当前配置及方向。所有输入源和推理参数在“设置”弹窗内调整，可切换内置测试视频或本地视频文件；点击“应用”生效，取消不改变当前配置。运行时应用后端、分辨率或输入源变更会重启推理，阈值变更直接生效。服务器只提供网页、模型、运行库和内置测试视频；视频帧在浏览器内处理，不上传到服务器。

在 **设置 → 摄像头设备** 中选择镜头并点击“应用”。列表和顶部当前摄像头名称来自浏览器；未提供名称时显示摄像头编号，不推测哪个是主摄或长焦。“自动”只请求优先后置，并不保证是主摄。明确选择时按 `deviceId` 精确打开，先释放旧镜头，运行中的推理随后恢复。选择保存在当前站点的 localStorage，刷新页面后继续使用；选择“自动”清除记忆。所选设备丢失或不可用时会提示重新选择，不静默切换。列表可以手动刷新，也会在授权和设备变化后更新；浏览器只开放一个逻辑摄像头时，页面无法列出其隐藏的物理镜头。

若需要重新导出模型：

```powershell
python -m pip install -r requirements-export.txt
python export_web.py
```

模型沿用 Python 原型的 `nvidia/segformer-b0-finetuned-ade-512-512`。导出脚本会验证 ONNX 结构及与 PyTorch 的数值一致性。

## GitHub Pages 静态部署

```powershell
npm ci
npm run build
```

`dist/` 是完整的静态站点，包含页面、Worker、ONNX Runtime、WASM、两个 ONNX 模型和内置测试视频。所有运行资源使用站点内的相对地址，不访问 CDN 或模型下载服务；支持根路径及 `https://用户名.github.io/仓库名/` 项目路径。发布的是整个 `dist/` 目录内容，不是单独一个 HTML 文件。

仓库已提供 `.github/workflows/pages.yml`：在 GitHub 的 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。向 `main` 或 `master` 推送后，工作流会安装构建依赖、生成 `dist/` 并部署。也可以在 Actions 手动运行该工作流。GitHub 构建只需要 Node.js，无需 Python；请连同 `web/models/*.onnx` 与测试视频提交，文件均低于 GitHub 单文件限制，不要将它们替换成未展开的 LFS 指针文件。

若使用分支部署，将 `dist/` 的**内容**复制到发布分支根目录，保留 `.nojekyll`，再在 Pages 设置中选择该分支的根目录。目录内的 `asset-manifest.json` 记录每项资源大小和 SHA-256。

GitHub Pages 提供 HTTPS，可以直接用于手机摄像头；浏览器仍会请求摄像头授权。Pages 不提供 COOP/COEP 响应头，因此 WASM 自动使用单线程，WebGPU 不要求这些响应头。无需 Service Worker 注入或特殊服务器配置。

静态资源总计约 96.6 MiB；运行库包含当前版本需要的 Asyncify WASM，以及隔离环境可用的 JSEP 文件。首次运行加载所选模型和实际需要的运行库，不会下载两个模型及所有后端。模型许可和运行库许可随 `licenses/`、`THIRD_PARTY_NOTICES.md` 一起打包；当前 SegFormer 上游许可限于非商业研究或评估。

本地模拟 Pages（无特殊响应头、无业务路由）：

```powershell
npm run preview
```

打开 http://127.0.0.1:8081。不要用 `file://` 双击运行，模块 Worker 和摄像头需要 HTTP localhost 或 HTTPS。

```powershell
npm run test:static
```

该检查在 `/pages-test/PRTS_Core/` 多级子路径运行，禁用对站点外资源的请求，验证无跨源隔离时的 WASM 单线程、WebGPU 两档模型、自动回退及摄像头。结果位于 `outputs/static/validation.json`。

## GPU 与 WASM

- **自动**：尝试 WebGPU 并实际执行预热推理，失败时自动切换 WASM。运行中的 GPU 推理异常也会触发回退。
- **WebGPU**：强制使用 GPU；不可用时显示错误，不会静默伪装成 GPU 运行。
- **WASM**：使用 CPU SIMD；满足跨源隔离条件时使用最多 4 个线程，否则单线程。

WASM 本身不是调用 GPU 的接口。本项目通过 ONNX Runtime Web 的 WebGPU 后端调用浏览器暴露的 GPU；浏览器及驱动决定底层使用的图形 API。它不直接调用 Qualcomm QNN、Apple Core ML 或设备 NPU。

| 设备 | 使用条件 |
|---|---|
| Qualcomm / Adreno Android | 浏览器、系统与驱动实际提供 WebGPU；未提供时回退 CPU |
| Apple Silicon Mac / iPhone / iPad | 使用提供 WebGPU 的浏览器及系统版本；不支持的版本回退 CPU |
| Intel / AMD / NVIDIA 桌面设备 | 浏览器启用硬件加速且 WebGPU adapter 可用 |

网页显示实际启用的后端、浏览器提供的 adapter 信息和实测帧率。GPU 后端可能仍将少量形状等运算放到 CPU。暂未在 Apple 或 Qualcomm 真机测试，不保证所有设备的实时帧率。

## 手机访问

服务器默认监听 `0.0.0.0:8080`，但手机访问 `http://电脑局域网IP:8080` 通常不满足安全上下文要求，不能打开摄像头或 WebGPU。需要 **手机信任的 HTTPS 证书**，或者将页面部署到 HTTPS 主机。

服务器支持 TLS 环境变量。准备好证书后，在 PowerShell 中运行：

```powershell
$env:TLS_CERT = 'D:\certificates\local-cert.pem'
$env:TLS_KEY = 'D:\certificates\local-key.pem'
$env:PORT = '8443'
npm start
```

在同一局域网的手机上打开 `https://电脑局域网IP:8443`。证书必须包含该 IP 或实际使用的域名，并受手机信任。可用 mkcert 等本地开发证书工具签发；根证书需在测试手机安装并信任。私钥不要分享或提交到仓库。浏览器证书警告页面不等于已经满足摄像头和 WebGPU 的要求。

使用反向代理或静态部署时，保留以下响应头以启用多线程 WASM：

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

静态部署直接使用 `npm run build` 生成的 `dist/` 内容；运行库和测试视频会复制到同一站点目录。开发模式的 Node 服务器继续提供同样的资源和视频 Range 请求。

## 处理逻辑

1. 摄像头请求不指定屏幕宽高比，并优先设置 `resizeMode: none`，避免浏览器为适应屏幕而裁切。预览、视频文件和采集帧均保留浏览器返回的完整画面，等比缩放，比例不一致时留边。模型输入也采用等比缩放加补边，再按 ImageNet 参数归一化。
2. Worker 内执行 SegFormer，返回地面概率和类别。
3. 将预测结果中的人工补边移除，仅在完整原图对应区域选择与近端中央连接的主要地面区域，在低分辨率掩码上收缩边缘、提取连续横截面的中心线。四周原始画面仍全部保留。
4. 根据前方中心点相对画面中央的位置，显示偏左、向前、偏右或暂无法判断。
5. 结果映射回完整采集帧；同一时刻只处理一帧，不积压视频队列。下一次读取最新视频帧。横竖屏、视口大小或视频原始尺寸变化时，立即清除旧叠加并丢弃之前的在途结果。页面进入后台或点击停止时终止 Worker、释放摄像头。

快速档输入为 192×320，精细档为 288×512。这是包括补边的模型输入大小，不是从原图中心截取的像素尺寸。例如 1280×720 横向帧会整体缩成 192×108，放入 192×320 模型输入，上下补边。预测移除补边后再映射回完整原图，预览及推理均不居中裁切或横纵不等比拉伸。浏览器版在 1/4 输入分辨率的预测网格上计算通道，横屏帧在竖向模型内占据的有效像素较少，分割边界可能更粗；需要细节时可选精细档。网页无法保证设备内部光学镜头与 ISP 完全无裁切，但页面自身不再裁切收到的画面。

性能显示中的“处理帧/秒”是实际结果输出频率；“单次推理”包含模型执行和输出读取；“端到端”包含采集、预处理、Worker、后处理及绘制，均不包括首次模型加载。

绿色表示预测地面，不代表身体净空或通行安全。箭头是图像内候选通道方向，不是从光流估计的实际运动方向。当前没有深度、距离、目的地规划或额外大模型逻辑。

## 验证

```powershell
npm test
npx playwright install chromium
npm run test:browser
npm run test:camera
npm run test:camera-selection
```

运行浏览器检查前保持 `npm start` 运行。浏览器测试覆盖 CPU、真实 WebGPU 推理、精细档、本地视频、模拟摄像头及停止释放、GPU 缺失自动回退、强制 GPU 报错恢复和手机尺寸布局。强制 GPU 测试需要测试机器提供 WebGPU；测试不使用强制开启 WebGPU 的实验参数。

`outputs/web/browser-validation.json` 保存首次性能测试数据，`outputs/web/ui-validation.json` 保存当前摄像头主界面的回归验证结果，`outputs/web/` 中保存桌面、手机及设置弹窗截图。界面检查覆盖默认摄像头预览、开始/停止、弹窗应用/取消、运行中配置更新及权限拒绝后的恢复。模拟摄像头用于验证生命周期，不代表已完成手机真机测试。

当前本机 Chromium / Intel Arc B390（adapter: intel xe-3lpg）的结果：快速档 WebGPU 260 帧持续输出约 49.7 FPS，平均推理 17.2 ms；精细档 20 帧约 28.5 FPS，平均推理 30.5 ms；快速档 WASM 90 帧约 12.2 FPS，平均推理 78.1 ms。统计跳过最初两帧。短片测试结果会受设备负载、温度及浏览器影响，不能外推为手机性能保证。

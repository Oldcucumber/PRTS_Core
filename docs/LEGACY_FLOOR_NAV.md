> 当前多模态核心交付入口：[BRAIN_PROTOTYPE](BRAIN_PROTOTYPE.md)、[最终验证](FINAL_VALIDATION.md)、[ZIP 使用说明](BRAIN_BUNDLE_README.md)。下文保留早期原型记录，不能用于替代新包的模型与资源结论。

# 地面识别与候选前进方向：Python 快速原型

新增离线后端 Demo：人行道与障碍共同规划候选通道，中文录音驱动导航、场景问答和持续等待，支持 JSONL、视频及答复 WAV。运行说明与实测边界见 [PRTS_CORE_README.md](../PRTS_CORE_README.md)。

在线测试：[GitHub Pages](https://oldcucumber.github.io/PRTS_Core/)。

另有浏览器测试网页：运行 `npm install`、`npm start` 后访问 http://localhost:8080。支持 WebGPU、WASM、测试视频与摄像头；设置中可开启 Depth Anything V2 Small 深度辅助，左右对比同一帧的“仅分割 / 分割＋深度”，默认关闭。深度输出是相对值，不是米制距离。用法和实测结果详见 [WEB_README.md](../WEB_README.md)。

GitHub Pages：`npm ci && npm run build` 生成包含全部运行资源的 `dist/`，仓库内已提供 Pages 自动部署工作流。使用 `npm run test:static` 验证纯静态环境。

以下说明对应 Python 原型，仅实现测试视频的地面分割和图像空间前进方向估计；可选深度辅助在浏览器版本中实现。

## 运行

```powershell
python -m pip install -r requirements.txt
python floor_nav.py VID20260919182406.mp4
```

首次运行从 Hugging Face 下载 SegFormer-B0 ADE20K 权重，后续使用本机缓存。默认 CPU，4 个线程。`--fps 8` 控制视频采样频率，不代表处理速度；输出视频维持近似原始时长。`--height 640` 控制输出高度，`--threshold 0.55` 控制地面概率阈值。

## 输出

- `outputs/annotated.mp4`：绿色地面、黄色中心线、蓝色候选方向箭头。
- `outputs/preview.jpg`：六个时刻的预览。
- `outputs/metrics.json`：逐帧方向、中心线、时间戳和实测性能。

## 方法与含义

SegFormer-B0 提取 floor 类别，概率阈值过滤后，选择与近端中央区域相接的主要连通区域。对掩码轻微腐蚀，在多条横向扫描线上选择连续地面区间，连接中点。中心线如果穿过非地面区域立即截断；缺少足够中心点时输出 UNKNOWN。根据前方中心点相对画面中央的位置输出 LEFT、FORWARD 或 RIGHT。

这是当前画面中候选通道的方向，不是相机或人的实际运动方向；未做视觉里程计、深度估计、目的地导航或人体宽度验证。绿色表示预测地面，不表示通行安全。当前不进行跨帧路径平均，避免镜头转向后延用旧路径；边界和方向仍可能抖动。反光、细小障碍及遮挡可能产生识别错误。

## 当前测试

测试源：VID20260919182406.mp4，1080×1920，278 帧，约 4.64 秒。
本机 CPU 全片解码、采样处理 40 帧；平均分割与路径处理耗时约 298ms（约 3.35 帧/秒），不含模型加载。输出采样帧率约 8.56fps，不代表实时推理达到该速度。已检查六时刻预览及输出视频完整解码；尚无人工逐帧标注，因此未声明分割精度。

"""Summarize frozen comparisons without converting descriptive counts to accuracy."""
import collections,hashlib,json,subprocess
from pathlib import Path
import numpy as np

ROOT=Path('outputs/stage2/perception')

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def totals(split):
    rows=read(ROOT/'path-comparison'/f'{split}-summary.json');result={}
    for tag in sorted({r['variant'] for r in rows}):
        chest=[r for r in rows if r['variant']==tag and r['camera']=='chest'];head=[r for r in rows if r['variant']==tag and r['camera']=='head']
        result[tag]={'chest_frames':sum(r['frames'] for r in chest),'chest_paths':sum(r['paths_valid_in_current_prediction'] for r in chest),
            'chest_confirmed_directions':sum(r['non_unknown_direction_frames'] for r in chest),
            'chest_direction_changes':sum(r['direction_changes_between_confirmed'] for r in chest),
            'current_mask_outside_pixels':sum(r['path_pixels_outside_current_prediction'] for r in rows if r['variant']==tag),
            'head_path_frames':sum(r['paths_valid_in_current_prediction'] for r in head),
            'head_path_pixels':sum(r['human_label_path_pixels'] for r in head),'head_path_violation_pixels':sum(r['human_label_path_violation_pixels'] for r in head)}
    return result

def main():
    dev,test=totals('development'),totals('heldout');base=read(ROOT/'comparison/b4-v1/heldout-summary.json')['human_head']
    selected=read(ROOT/'comparison/mask2former-fp16/heldout-summary.json')['human_head'];a=test['fusion-temporal'];b=test['v1']
    gates={'walkable_iou_gain_at_least_0_03':selected['aggregate_walkable_iou']-base['aggregate_walkable_iou']>=.03,
        'fixed_admission_no_worse':selected['fixed_admitted_pixels']<=base['fixed_admitted_pixels'],
        'dynamic_admission_no_worse':selected['dynamic_admitted_pixels']<=base['dynamic_admitted_pixels'],
        'zero_path_pixels_outside_current_prediction':a['current_mask_outside_pixels']==0,
        'human_path_violation_rate_no_worse':a['head_path_violation_pixels']/a['head_path_pixels']<=b['head_path_violation_pixels']/b['head_path_pixels']}
    frozen=read(ROOT/'acceptance-freeze.json');gates['frozen_sources_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen['source_sha256'].items())
    quant=read(ROOT/'cpu-validation/int8/results.json');float32=read(ROOT/'cpu-validation/fp32/results.json');low=read(ROOT/'cpu-validation/fp32-low-memory/results.json')
    agreement=float(np.mean([r['class_agreement'] for r in quant['rows']]))
    qm=read(ROOT/'onnx/mask2former-int8/manifest.json');qm.update(passed=False,validation_status='rejected_by_predeclared_raw_class_agreement',
        mean_raw_class_agreement=agreement,required_mean_raw_class_agreement=.98,
        note='Most mismatch on head frame 36 is Pedestrian Area vs Sidewalk, both walkable; this is fine-label instability, not a collapse of walkable ground. Threshold was not revised after observing it.')
    (ROOT/'onnx/mask2former-int8/manifest.json').write_text(json.dumps(qm,indent=2),encoding='utf-8')
    videos=[]
    for name in ['source-time','processing-time']:
        p=ROOT/'continuous-turn'/(name+'.mp4')
        probe=json.loads(subprocess.check_output(['D:/ffmpeg/bin/ffprobe.exe','-v','error','-count_frames','-show_entries','stream=width,height,avg_frame_rate,nb_read_frames,duration','-of','json',str(p)]))
        subprocess.run(['D:/ffmpeg/bin/ffmpeg.exe','-v','error','-i',str(p),'-f','null','-'],check=True)
        videos.append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'probe':probe,'fully_decoded':True})
    (ROOT/'acceptance-results.json').write_text(json.dumps({'frozen_gates':gates,'development':dev,'heldout':test,'media':videos},indent=2),encoding='utf-8')
    model_rows=[]
    for tag in ['b4-v1','b4-aspect','yolo26s','yolo26m','mask2former','mask2former-fp16']:
        d=read(ROOT/'comparison'/tag/'development-summary.json');tp=ROOT/'comparison'/tag/'heldout-summary.json';t=read(tp) if tp.exists() else None
        model_rows.append(f'| {tag} | {d["human_head"]["aggregate_walkable_iou"]:.4f} | {t["human_head"]["aggregate_walkable_iou"]:.4f} |' if t else f'| {tag} | {d["human_head"]["aggregate_walkable_iou"]:.4f} | 未重复运行 |')
    path_rows=[]
    for tag in dev:
        d,t=dev[tag],test[tag]
        path_rows.append(f'| {tag} | {d["chest_paths"]}/226 | {d["chest_direction_changes"]} | {t["chest_paths"]}/240 | {t["chest_direction_changes"]} | {t["head_path_violation_pixels"]}/{t["head_path_pixels"]} |')
    memory=read(ROOT/'cold-cuda.json');media=read(ROOT/'continuous-turn/manifest.json')
    report=f'''# 第二阶段：感知与局部路径子任务记录

研究候选采用 **Mask2Former Mapillary + YOLO11n + 连通路径与转向滞回**。独立会话的离线比较达到预先冻结的感知/路径像素门槛；全部应用的地图、语音调度、8 GB预算、Apple构建由总任务继续集成验收。本报告不把局部研究完成算成整个项目完成。

## 输入与证据边界

- SANPO-Real，CC-BY-4.0。12个会话按会话分成6个开发、6个独立验收；所有来源URL、原PNG SHA-256、相机名、帧时刻和标注类型保留在 `data/*/manifest.json`。
- 共601张胸前原始帧，其中 dev-turn 为180张无缺帧、15 FPS、12秒连续画面，其余为每4帧采样，原始时间轴约5.6–10.7秒。比较使用226开发胸前帧、240独立胸前帧。
- **胸前图像没有公开分割真值**。像素指标另用匹配头部相机的8张开发、48张独立人工标注；连续帧相关，不能当作56个独立场景。标签RGB红通道为语义ID，0忽略，3/6/17作为walkable。头部真值从未配到胸前图像。
- 这些会话未用于本阶段训练；本阶段没有训练。独立会话未用于第一版本地SANPO适配。预训练数据与公开场景的重叠未审计。
- 已下载官方修正的 `fixed_camera_poses.csv`，本次没有消费它生成世界路径；当前路径与视觉运动均为图像坐标，未验证米制净空或AR空间投影。
- `acceptance-freeze.json` 在独立模型推理前保存参数、源码SHA和门槛。当前4份冻结源码哈希仍一致：{gates['frozen_sources_unchanged']}。

## 同输入模型比较

| 配置 | 开发头部 walkable IoU | 独立头部 walkable IoU |
|---|---:|---:|
{chr(10).join(model_rows)}

B4第一版为512平方拉伸；B4-aspect、YOLO26s/m、Mask2Former为1024长边，实际576×1024。分辨率/输入几何同时变化，不把全部差异归因于架构。YOLO26使用真正的Cityscapes密集语义模型，RGB/255预处理与其本地官方predictor一致；广场、步道常判成road。B4改输入几何也没有稳定提升。Mapillary类别更贴近这批行走场景，但不代表在所有地区优于其他模型。

选中模型独立IoU比V1提高 **{selected['aggregate_walkable_iou']-base['aggregate_walkable_iou']:.4f}**。误纳入固定实体像素为 **{selected['fixed_admitted_pixels']} vs {base['fixed_admitted_pixels']}**，动态物体为 **{selected['dynamic_admitted_pixels']} vs {base['dynamic_admitted_pixels']}**，路面/过街类为 **{selected['roadway_admitted_pixels']} vs {base['roadway_admitted_pixels']}**。这些是180×320评测网格像素计数。

Mask2Former加载警告已审计：把缺失的Swin汇总layernorm参数改成权重7/偏置-11，分割class/mask logits最大差均0；中间feature maps不依赖该输出。旧relative_position_index在当前Transformers里按window_size重算且不持久化。见 `model-audit/audit.json` 与 `scripts/audit_perception_models.py`。

## 路径与方向消融

| 组合 | 开发有效路径 | 开发方向切换 | 独立有效路径 | 独立方向切换 | 独立人工标签路径越界像素 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(path_rows)}

新时序版独立240帧只有 **{a['chest_paths']}条候选路径、{a['chest_confirmed_directions']}次确定方向**。开发216条路径中199次确定方向，其余17次等待方向确认；独立79条中4次暂缓方向确认。切换少不是方向准确率，未获得独立人工的逐帧正确转向标签，不能宣称真实转向准确率已达标。

路径改动：从同一近场锚点出发，用4连通分量限制候选、8邻接A*寻找弯曲通道，禁止切角；每一段完整栅格化验证，像素边距是启发式而非人体宽度。方向用固定近/远行的切线趋势，不再取随路径长度变化的75%点。旧路径只在视觉仿射注册质量足够时提供弱代价偏好，不能扩大当前free mask；当前通道失效立即取消。滞回期间输出UNKNOWN，不继续保持反向旧箭头；较长推理间隔清空历史，但仍能使用新一帧有效几何。

所有新路径在**自身预测**free mask内越界0；外部人工标注仍有 **{a['head_path_violation_pixels']}/{a['head_path_pixels']}** 像素落到非walkable类别，不能声称路径绝对正确。语义单模型/加入检测/加入时序分别保留完整JSONL。`fusion-no-temporal`关闭注册偏好和方向滞回，仍保留输入时刻与极端画面运动检查。

Guide兼容原调用，增加 `route_hint=None`。只在following且有可靠 `relative_bearing_deg` 时作为左右偏好；缺失相机朝向不投影，视野外单独标记，明显反向的局部通道返回WAIT/route_corridor_conflict。图像heading不等于罗盘朝向或人的速度。语义regions保留原始类别、组、轮廓与孔洞；细小区域会省略、最多64区域，仅供展示，dense mask才是路径依据。植被未改名为花坛。

## 便携运行与资源

- `onnx/mask2former-1024/semantic.onnx` 实际导出，输入RGB float32 NCHW [1,3,576,1024]并按ImageNet归一化，输出65类概率[1,65,180,320]。导出器把GridSample坐标升为double；为18处采样坐标增加float32 Cast后，ORT CPU加载和推理成功。两张真实图与FP32 PyTorch掩码100%一致，概率最大误差7.23e-5。初次两图约5.7–6.0秒/帧。
- `onnx/yolo11n-rect/detector.onnx` 为[1,3,384,640]，RGB/255、114填充、xywh+80类分数，CPU实际验证；使用相同预处理时的归一化框误差<1e-4、类别分数误差<1e-4、0.25阈值候选一致。原square640导出作为实验记录保留，集成使用rect版。
- `ONNXPerception`只依赖numpy、OpenCV、onnxruntime；实际CPU进程确认未导入torch。可处理letterbox，当前数值验证范围为16:9源图；其他纵横比/旋转须另行设备验证。正式路径没有TensorRT依赖。
- FP32 ONNX在14个开发真实输入上的峰值Windows工作集 **{float32['peak_working_set']/1e9:.3f} GB**；`low_memory=True`禁用CPU arena、memory pattern和预打包后为 **{low['peak_working_set']/1e9:.3f} GB**，运行后RSS **{low['final_rss']/1e9:.3f} GB**。14张分类掩码均与原配置100%一致，降低占用伴随延迟变化，仍超感知1.2 GB初始分配。
- 冷启动CUDA研究进程：峰值RSS **{memory['peak_process_working_set']/1e9:.3f} GB**，CUDA allocated **{memory['peak_cuda_allocated']/1e9:.3f} GB**、reserved **{memory['peak_cuda_reserved']/1e9:.3f} GB**。保守RSS+reserved为 **{memory['conservative_peak_rss_plus_reserved_bytes']/1e9:.3f} GB**，包括Python/库，CUDA计数不含全部驱动上下文；这不等同Apple统一内存footprint。完整应用仍须测同时峰值。
- 动态INT8 MatMul/Gemm导出为273 MB，但14张平均原始类别一致度 **{agreement:.4f}**，低于预设0.98，**未采用**。最低一帧主要Pedestrian Area→Sidewalk，两者都walkable，五分组一致99.82%，不是地面崩坏；保留这个细类稳定性失败，不改门槛放行。卷积仍FP32。
- Core ML转换、Mac/Xcode构建、Apple真机性能及热状态均未执行。

## 视频与具体失败

完整胸前连续序列：`continuous-turn/source-time.mp4`，180帧、12秒；同一次实测比较处理节奏：`continuous-turn/processing-time.mp4`，{media['processing_video_duration_s']:.2f}秒。源时间版本明确标出相对比较处理的{media['source_video_processing_acceleration']:.2f}倍加速，两个模型与绘图开销合计，文件编码不计。新管线在这次比较中的p50={media['new_processing_p50_ms']:.0f}ms、p95={media['new_processing_p95_ms']:.0f}ms，不能当手机速度。视频已ffmpeg全量解码验证；JSONL逐帧保留输入SHA、源/处理时间、方向、路径和原因。选定7张截图可直接检查，处理视频不包含语音；语音链路由总任务另行验收。

- test-park、test-rural 的林间/乡间地面多数保留unknown，test-rural 40/40 WAIT；不把缺少类别支持的地面直接改成可走。
- test-park-b 的相机常抬高，近场可见地面不足，39/40 WAIT。
- test-urban-b 视角偏高、初段行人遮挡锚点，40/40 WAIT。头部标注IoU好不意味着同会话胸前视角也可导航。
- 检测框仍可能误识别或过度阻断，语义边缘也会错；人工标签中的2个路径越界像素和所有失败行保留。广场多个通道同时可走，单一左右标签也没有独立真值。
- 现有宽度余量是图像启发式，不支持米数、真实人体净空、可过街或登车许可。后续要结合团队相机安装角度、标定/位姿/深度与实际胸前录制再验收。

## 接入与复现

研究构造：`Perception(segmentation='mask2former-mapillary', long_side=1024, square=False, precision='fp16', device='cuda')`；便携参考：`ONNXPerception('outputs/stage2/perception/onnx/mask2former-1024', 'outputs/stage2/perception/onnx/yolo11n-rect', low_memory=True)`。未修改Perception历史默认配置或demo调用；总任务显式接入选中配置。

在工作区使用 `.venv-research/Scripts/python.exe`：

```powershell
python scripts/prepare_perception_data.py
python scripts/compare_perception_stage2.py --split development
python scripts/compare_perception_stage2.py --split heldout --models b4-v1 b4-aspect yolo26s yolo26m mask2former-fp16
python scripts/replay_perception_comparison.py --split development
python scripts/replay_perception_comparison.py --split heldout
python scripts/render_perception_sequence.py
python scripts/label_perception_video.py
python scripts/measure_perception_runtime.py
python scripts/validate_quantized_perception.py --model outputs/stage2/perception/onnx/mask2former-1024 --tag fp32
python scripts/validate_quantized_perception.py --model outputs/stage2/perception/onnx/mask2former-1024 --tag fp32-low-memory --reference fp32 --low-memory
python -m unittest discover -s tests -p test_offline_core.py
python -m unittest discover -s tests -p test_perception_stage2.py
python scripts/report_perception_stage2.py
```

模型下载与导出命令见 `REPRODUCE.md`。依赖沿用研究虚拟环境：torch2.12.1+cu126、transformers5.17.0、ultralytics8.4.77、onnxruntime1.23.2、opencv4.10、numpy1.26。CPU推理不需要torch，但导出需要。11项旧回归+9项新几何/区域测试通过；这些构造测试不当感知准确率证据。

用户要求公交线路泛化：`Brain.intent`已去掉912/A108/二号线固定实体例子，改成提取本次用户标识、保留字母与前导零、缺失留空的通用规则。未加入视频/线路分支。多线路公交和方向绑定回归由总任务执行。

第一版 `outputs/final-*`、`outputs/review` 与 `baseline-v1` 未改动。未提交、推送或部署。

## 来源与许可

- [SANPO官方数据说明](https://github.com/google-research-datasets/sanpo_dataset)：相机、标签、修正位姿、CC-BY-4.0。
- [YOLO语义分割官方说明](https://docs.ultralytics.com/tasks/semantic)：密集语义模型及训练数据。YOLO代码/权重许可按Ultralytics发布条款与随包清单处理。
- [Mask2Former Mapillary权重](https://huggingface.co/facebook/mask2former-swin-large-mapillary-vistas-semantic)、[官方模型库许可](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md)：预训练权重CC-BY-NC-4.0，需要随集成保留许可与来源，不能按无条件商业权重分发。
'''
    (ROOT/'REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps(gates,indent=2));print('Report generated; full task acceptance remains with the root task.')

if __name__=='__main__':main()

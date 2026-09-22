"""Summarize actual frozen model results and a complete measured replay."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))

def main():
    model=read('outputs/stage2/brain/final-v9-q8-t6/results.json')
    buses=read('outputs/stage2/brain/final-bus-v9-q8-t6/results.json')
    run=read('outputs/stage2/streaming/brain-final/summary.json')
    events=[json.loads(x) for x in (ROOT/'outputs/stage2/streaming/brain-final/events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert model['complete'] and buses['complete']
    lines=['# 最终多模态快速原型验证', '',
        '本报告对应包内真实模型及当前代码。它证明桌面后端链路能够运行，并保留已知失败；不等同于全部场景或 iPhone 产品验收。', '',
        '## 固定配置', '',
        '- 大脑：官方 Qwen3-VL-4B-Instruct，语言 Q4_K_M、视觉 Q8_0；goal-scene-v9；greedy；6 CPU 线程；4096 context。',
        '- 原生库：llama.cpp b29c606e28a01b1bc8c1351026a0fa6e616bf6c4；关闭额外 CPU repack 权重副本。',
        '- 小脑：Mask2Former Mapillary FP16 v2 + YOLO11n，ONNX Runtime DirectML；SenseVoice INT8；原始 PP-OCRv6；Windows Huihui TTS。',
        '- 实测电脑：i5-13600KF、RTX 4070 Super 12 GB、32 GB RAM；视觉使用 DirectML，大脑使用 CPU，无 CUDA/Torch 依赖。',
        '- 原型依靠有限历史与最新帧调度形成持续观察，不是原生全双工音视频模型；运行时不训练权重。', '',
        '## 真实大脑推理', '',
        '| 冻结用例 | 正例提醒 | 负例误提醒 | 不确定输出 | 总用例 |', '|---|---:|---:|---:|---:|']
    for title,r in [('场景/编号角色/否定/时序',model),('保留公交线路与电子标识变化',buses)]:
        lines.append(f"| {title} | {r['positive_hits']}/{r['positive_count']} | {r['false_positives']}/{r['total']-r['positive_count']} | {r['uncertain']} | {r['total']} |")
    lines+=['', '正例分母表示应提醒的用例，负例分母表示不应提醒的用例。负例未提醒可能是正确识别，也可能是看不清，不能把二者合成“识别准确率”。完整标识和事件类型检查可以拒绝模型内部不一致的 MATCH，原始 model_decision 与最终 decision 同时保留。', '',
        '这批照片/录像是已检查过的开发与保留素材，不是独立盲测。18 项公交测试在本次候选推理前冻结，包含 20 路不同时刻的电子标识、466、8、20A、SL3、73 和用户的 912 录像，及配对错误目标；帧选取曾参考旧 OCR 实验，不能声称完全无选择偏差。三个广播用例输入的是脚本转写，不能作为真实广播 ASR 识别率。', '',
        '### 未命中或误提醒（完整保留）', '']
    for title,r in [('场景',model),('公交',buses)]:
        failed=[x for x in r['results'] if not x['passed']]
        if not failed:lines.append(f'- {title}：本组未出现最终任务判断错误；仍需参考不确定输出。')
        for x in failed:
            c=x['case'];v=x['response']
            lines.append(f"- {title} `{c['id']}`，目标 `{c['target']}`，期望提醒={c['expected']}，得到 {v['decision']}；观察：{v['observation']}。")
    lines+=['', '### 选型比较', '', '| 模型/版本 | 正例提醒 | 误提醒 | 范围 |', '|---|---:|---:|---|']
    for folder,title in [('qwen3vl2b-native-v9-q8-t6','Qwen3-VL 2B / v9'),('qwen3vl-native-v8-f16','Qwen3-VL 4B / v8 / F16 视觉'),('qwen4b-native-matrix-v7','Qwen3.5 4B / v7')]:
        path=ROOT/'outputs/stage2/brain'/folder/'results.json'
        if not path.exists():continue
        r=json.loads(path.read_text(encoding='utf-8'))
        lines.append(f"| {title} | {r['positive_hits']}/{r['positive_count']} | {r['false_positives']} | {r['total']}，完成={r['complete']} |")
    lines+=['', '2B 曾将车身编号、站牌、柜台号、非当前叫号和价格错当目标，没有为节省内存采用它。更早 MiniCPM 及其他提示词的原始失败仍在 outputs/stage2/brain。不同提示词/量化/运行参数的对比不是严格消融实验。', '', '## 连续交互与资源', '']
    lines+=['| 发生时刻（秒） | 事件 | 实际输出 |','|---:|---|---|']
    for e in events:
        if e['type'] in ('target_observed','answer','wait_cancelled','transcript'):
            text=e.get('text','').replace('|',' / ').replace('\n',' ')
            lines.append(f"| {e['emitted_s']-run['started_monotonic_s']:.2f} | {e['type']} | {text} |")
    lines.append('')
    stats=run['stats'];memory=run['memory'];notices=[e for e in events if e['type']=='target_observed']
    errors=[e for e in events if e['type'] in ('worker_error','request_error')]
    lines += [f"- 输入 {run['source_end_s']-run['source_start_s']:.0f} 秒，{stats['frames_received']} 帧，处理 {stats['frames_processed']} 帧，替换 {stats['frames_replaced']} 帧，运行错误 {len(errors)}。",
        '- 确认提醒：'+ '、'.join(e['wait']['target'] for e in notices)+'。柜台 3 为反例；B003 取消后重新创建任务。',
        '- 临时问答的 source 必须为 local_vision_language_model；实际文本、截断与播报时间保存在事件日志中。',
        f"- 内存：{memory['samples']} 次采样；进程树 RSS 峰值 {memory['peak_process_tree_rss_bytes']/1e9:.3f} GB；RSS + 同时刻 DirectML local/nonlocal 保守包络峰值 {memory['peak_conservative_desktop_envelope_bytes']/1e9:.3f} GB。",
        f"- 加假设的 2 GB 前端预留后为 {memory['with_assumed_frontend_2gb']/1e9:.3f} GB。8 GB 整套预算满足={memory['with_assumed_frontend_2gb']<=8_000_000_000}；不能用权重体积或旧 OCR 版本峰值替代。",
        '- DirectML shared/nonlocal 与 RSS 可能重复计算，包络是保守值；前端 2 GB 没有实测。Windows 数值不能外推成 Apple 统一内存峰值。',
        '- CPU 大脑单次视觉分析约数十秒，提示包含证据年龄；这仍是公交移动目标的明显实用性限制。采用小模型会降低延迟，但本轮语义错误不满足选择要求。',
        '- 播放调度复核：三个目标提示均完整播放；视觉答复完整生成，9.70 秒的语音在第 246 秒新用户命令到来时被中断，已播放 6.44 秒。不是被相同导航状态的重复提示打断；完整 WAV 随包保留。',
        '- 录像使用实际公共照片与 SANPO 胸前道路片段拼接、脚本命令与合成用户 PCM；系统实际运行模型与 ASR/TTS。音频按生成可用时间重建播放，非物理扬声器测量。', '',
        '## 地面、地图与迁移', '',
        '- 保留地面效果比较：Mask2Former 头戴人工独立切分的可走 IoU 0.9178，旧 B4 为 0.7962；胸前保留集仅 79/240 帧产生候选路径，75 帧确认方向，不能声称普遍可用。详见 perception/REPORT.md。',
        '- 地图录像来自实际高德 6 步、约 1250 米路线，627 个 GPS 位置为仿真。它验证路线状态/目的地确认/转弯，不证明视频与地图已做世界空间 AR 注册。',
        '- 59 项 Python 控制流测试通过；23 个 Swift 文件通过语法树检查。控制流使用模型替身，不计入实际模型效果；没有 Xcode 编译、Metal/CoreML 或 iPhone 性能实测。',
        '- LiDAR、深度、AR 位姿为接口预留；米制避障、实地定位/朝向标定、实时信号灯通行决策、真实地铁等待和全新未见场景仍需后续验证。', '',
        '## 复核入口', '',
        '模型用例及逐项原始响应：outputs/stage2/brain/final-v9-q8-t6/results.json、final-bus-v9-q8-t6/results.json。连续回放：outputs/stage2/streaming/brain-final。图文录像：outputs/stage2/demos/brain-final-timeline。源码与模型 SHA256 随 BUNDLE.json；独立解压后的复核另交付 extraction-validation.json。', '',
        '下一阶段优先处理远处/变化标识的视觉取样与证据时延，在保留失败反例的固定回归上评估；随后压缩感知激活与 KV 并做 Apple 真机内存、定位和持续运行验证。当前待解决项没有用规则替换大脑或更改评分口径来隐藏。']
    (ROOT/'docs/FINAL_VALIDATION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Wrote docs/FINAL_VALIDATION.md from complete actual runs')

if __name__=='__main__':main()

"""Render the implemented queue evidence flow, including known unconfirmed cases."""
from render_architecture import diagram

diagram('04-queue-evidence','PRTS Core · 叫号如何产生提醒',
        '号码来自当前 OCR / ASR；用户目标只用于最后匹配。模型猜测、柜台号及未确认栏目不触发到号提醒。',{
    'camera':(.5,6.8,3.15,1.75,'真实画面与时钟','相机帧 / 序号 / 观测时间\n当前演示用真实照片时序拼接','blue'),
    'ocr':(4.15,6.55,3.6,2.15,'文字与可见几何','有界 OCR → 数字行上下文重读\n透视校正 → 检查字形数量\n按明显像素间距分组，不定长切割','green'),
    'fields':(8.3,6.55,3.1,2.15,'关联局部栏目','沿文字基线判断上下/左右关系\n叫到 / 等待 / 柜台 / 其他\n不确定和混合字段保留为未确认','green'),
    'decision':(12,4.8,3.4,3.1,'任务匹配与证据','字母 + 前导零 + 完整数字\n例：C-002 → C002，不能当 002\n\n已叫栏目或明确广播 + 完整匹配\n任务未被取消/替换 → 提醒一次\n记录证据文字、时间与输入帧','green'),
    'audio':(.5,3.9,3.15,1.75,'连续广播 / 用户语音','16 kHz PCM → 本地 ASR\n区分广播与用户命令','blue'),
    'audio_role':(4.15,3.65,3.6,2.15,'广播的服务对象','“请 A108 到 3 号窗口”\n提取 A108；3 是目的柜台\n否定、询问、过号不确认\n没听到的字母不能补写','green'),
    'target':(.5,1.1,7.25,1.75,'用户的等待任务','设定 / 修改 / 取消完整目标；与导航和问答并存\n分句尾部“到时提醒我”只延续当前任务，不重启已提醒任务','blue'),
    'output':(12,1.1,3.4,2.0,'输出给前端','target_observed / 候选事件\n文字 → TTS + 提示音\n画面字段、证据和时间线可回放','blue'),
    'limits':(9.0,1.1,2.4,2.0,'当前保留的失败','两种 LED 栏目未确认\nASR 可能漏掉字母\nApple 尚未编译实测','amber'),
},[('camera','ocr',''),('ocr','fields',''),('fields','decision','视觉证据'),
    ('audio','audio_role',''),('audio_role','decision','广播证据'),
    ('target','decision','目标只在此匹配'),('decision','output','')],
    '实验公开保留首轮误报和漏报；修正后的同一批样本属于开发回归。\n'
    'C++ 仅移植像素间距拆分；Swift 源码和桌面运行证据不能代替 Apple OCR / 真机验证。')

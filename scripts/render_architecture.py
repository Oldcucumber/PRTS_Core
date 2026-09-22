"""Generate editable SVG and PNG engineering diagrams from the implemented contract."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/diagrams'
FONT=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams['svg.fonttype']='none'


def diagram(name,title,subtitle,nodes,edges,notes):
    fig,ax=plt.subplots(figsize=(16,10),dpi=150)
    fig.patch.set_facecolor('#f5f7fa');ax.set_facecolor('#f5f7fa')
    ax.set_xlim(0,16);ax.set_ylim(0,10);ax.axis('off')
    ax.text(.5,9.55,title,fontproperties=FONT,fontsize=24,color='#14263e',weight='bold')
    ax.text(.5,9.07,subtitle,fontproperties=FONT,fontsize=11,color='#53657d')
    for source,dest,label in edges:
        a,b=nodes[source],nodes[dest];x1,y1=a[0]+a[2]/2,a[1]+a[3]/2;x2,y2=b[0]+b[2]/2,b[1]+b[3]/2
        if abs(x2-x1)>abs(y2-y1)*1.7:
            x1+=a[2]/2*(1 if x2>x1 else -1);x2-=b[2]/2*(1 if x2>x1 else -1)
        else:
            y1+=a[3]/2*(1 if y2>y1 else -1);y2-=b[3]/2*(1 if y2>y1 else -1)
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,color='#537798',linewidth=1.5,zorder=1))
        if label:ax.text((x1+x2)/2,(y1+y2)/2,label,fontproperties=FONT,fontsize=8.5,color='#365b78',ha='center',
                         bbox=dict(facecolor='#f5f7fa',edgecolor='none',pad=1),zorder=4)
    for key,(x,y,w,h,heading,body,color) in nodes.items():
        fill={'blue':'#e7f0fa','green':'#e4f3ef','amber':'#fff1d9','gray':'#edf0f4'}[color]
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04,rounding_size=.09',facecolor=fill,edgecolor='#aabccd',linewidth=1,zorder=2))
        ax.text(x+.16,y+h-.24,heading,fontproperties=FONT,fontsize=12.5,color='#14263e',va='top',weight='bold',zorder=3)
        ax.text(x+.16,y+h-.64,body,fontproperties=FONT,fontsize=10,color='#334a63',va='top',linespacing=1.65,zorder=3)
    ax.text(.5,.22,notes,fontproperties=FONT,fontsize=9.3,color='#53657d',va='bottom',linespacing=1.6)
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',pad_inches=.15)
    fig.savefig(OUT/(name+'.png'),bbox_inches='tight',pad_inches=.15,dpi=160)
    plt.close(fig)


def main():
    diagram('01-core-flow','PRTS Core · 连续感知与交互','实线为已实现的 Python / 原生组合；Apple 真机性能尚未测量。',{
      'camera':(.5,6.9,3.1,1.65,'摄像头输入','朝向统一的图像 + 单调时钟\n帧序号 / 可选空间信息','blue'),
      'vision':(4.25,6.65,4.1,2.15,'持续视觉工作线程','Mask2Former：地面 / 道路 / 实体\nYOLO11n：人、车、信号灯等目标\n连通区域 → 局部路径 → 方向稳定','green'),
      'scene':(9,6.9,3,1.65,'结构化场景','分区、检测框、图像路径\n观测时刻 / 耗时 / 有效期','green'),
      'audio':(.5,4.05,3.1,1.7,'连续音频输入','16 kHz 单声道 PCM\n角色 / 回声处理 / 分句','blue'),
      'speech':(4.25,4.05,4.1,1.7,'独立语音与交互线程','本地 ASR → 控制词 / 意图\n即时问答 / 持续等待 / 目的地确认','green'),
      'vlm':(9,4.05,3,1.7,'按需视觉语言推理','MiniCPM-V 4.6\nQ5 语言；视觉压缩另做对照\nC ABI：CPU / Metal 路径','green'),
      'maps':(12.6,6.65,2.8,1.9,'地图适配器','高德 POI / 步行路线\n明确选择后规划\n在线查询，沿线进度本地算','blue'),
      'state':(4.25,1.3,4.1,1.85,'任务状态与事件调度','导航与等待状态分开保留\n控制指令取消旧生成 / 旧播报\n同车证据、目标替换、提醒去重','green'),
      'output':(9,1.3,6.4,1.85,'输出到团队前端','文字 → 离线系统 TTS\n提示音指令 / 区域和路径 / 任务事件\n前端负责播放、渲染、采集及回声消除','blue'),
    },[('camera','vision','只保留\n最新帧'),('vision','scene',''),('audio','speech',''),('speech','vlm','按需'),
        ('speech','state',''),('scene','vlm','当前图像证据'),('vlm','state','带版本的结果'),
        ('maps','scene','路线\n方向偏好'),('state','output','优先级\n+ 有效期')],
      '范围：模型推理可离线；POI / 新路线查询需要地图服务。Python 是研究与回放参考，原生 VLM 库已在 Windows CPU 实跑。\n图像路径尚不是经过世界坐标标定的 AR 导航线；LiDAR / ARKit 输入接口预留，当前不依赖。')
    diagram('02-waiting-evidence','PRTS Core · 公交等待如何形成提醒','线路提取不输入“期待的号码”；匹配发生在证据提取之后。',{
      'intent':(.5,6.75,3.2,1.7,'用户提出等待目标','保留完整标识：20A / SL3 / 008\n可选方向；支持替换和取消','blue'),
      'detect':(4.35,6.75,3.2,1.7,'检测车辆与裁切','每条 OCR 绑定车辆观测 ID\n保留文字框及原始识别分数','green'),
      'ocr':(8.2,6.75,3.2,1.7,'OCR 读取候选文字','车上文字 → 显示区域候选\n区域位置仅是启发式','green'),
      'model':(12.05,6.75,3.2,1.7,'视觉模型独立复核','仅看这辆车的裁切图\n读线路，区分车身编号','green'),
      'match':(4.35,3.8,7.05,1.85,'证据核对与匹配','完整标识匹配；数字前后缀不能省略\nOCR 与视觉模型一致才形成确认观测\n指定方向时，方向文字必须属于同一辆车','green'),
      'candidate':(.5,1.1,4.1,1.8,'信息不足：target_candidate','只提示可能目标或方向未确认\n等待任务仍保留，不播放找到目标音','amber'),
      'confirmed':(5.95,1.1,4.1,1.8,'证据满足：target_observed','输出目标 / 来源 / 观测时刻\nTTS + 提示音；一次提醒后去重','green'),
      'negative':(11.4,1.1,3.85,1.8,'错误目标 / 旧证据','站牌、邻车、车牌、车身号\n旧广播、取消后的结果不触发','gray'),
    },[('detect','ocr',''),('ocr','model','独立读取'),('intent','match','用户条件'),('ocr','match','文字与位置'),
        ('model','match','线路语义'),('match','candidate','尚未确认'),('match','confirmed','一致且匹配'),('match','negative','不匹配')],
      '真实开发反例：20A 被旧规则截为 20；远距 8 路漏读；4833 车身编号误报。修复效果与未解决项均保存。\n独立测试按素材 / 线路分组。静态照片只证明标识识别；抽帧回放不代表到站响应速度或全天候可用性。')
    diagram('03-route-coordinates','PRTS Core · 地图路线如何约束局部路径','地图给出目的地与转向意图；相机判断眼前哪些区域可形成候选通道。',{
      'destination':(.5,6.7,3.25,1.8,'目的地确认','口语地点 → POI 候选列表\n用户选择 → 真实步行路线','blue'),
      'position':(4.3,6.7,3.25,1.8,'位置与路线进度','WGS84 / GCJ-02 明确区分\n定位精度、时间、沿线距离','blue'),
      'hint':(8.1,6.7,3.25,1.8,'近期路线意图','距下一动作多远 / 左右转\n仅可靠相机朝向产生相对角度','green'),
      'image':(.5,3.75,3.25,1.85,'图像语义分区','可走 / 道路 / 固定实体\n动态物体 / 未知\n不以框中空白推断可通行','green'),
      'path':(4.3,3.75,7.05,1.85,'局部通道与指向','近场连通性 → 障碍排除 → 图像栅格路径\n路径每段均检查当前预测可走区域\n路线意图只在已有可达通道间提供偏好','green'),
      'overlay':(12,3.75,3.35,1.85,'当前输出','归一化图像坐标的路径\n方向箭头 / 分区与目标\n缺乏证据时 WAIT / UNKNOWN','blue'),
      'future':(4.3,1.05,7.05,1.85,'预留：Apple 空间观测','相机内参、camera-to-world、深度、置信度、网格\n共享单调时钟、米制单位、世界坐标重置标识\n完成标定及定位验证后才生成世界空间路径','amber'),
    },[('destination','position',''),('position','hint',''),('hint','path','朝向可靠时'),('image','path','当前可走区域'),
        ('path','overlay',''),('future','path','未来适配接口')],
      '已实测：真实高德 POI / 步行路线调用，局部感知与路径对照。未实测：地图与实拍连续位置轨迹的世界空间对齐。\n不能把图像中的像素距离当作米数，也不能把红绿灯 / 车辆观测自动解释为获准通行。')


if __name__=='__main__':main()

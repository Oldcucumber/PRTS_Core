"""Render real AMap route geometry with explicitly simulated location updates.

This exercises destination confirmation, route progress and arrival-to-wait.
It does not simulate camera evidence, street imagery or AR spatial registration.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from prts_core.navigation import (Point,Location,Destination,Navigation,parse_amap_walking,bearing)
from prts_core.intent import route as parse_intent
from prts_core.tasks import Tasks


def main():
    out=ROOT/'outputs/stage2/maps/route-replay';out.mkdir(parents=True,exist_ok=True)
    initial=json.loads((ROOT/'outputs/stage2/maps/amap-live-route-neighborhood.json').read_text(encoding='utf-8'))
    audit_path=ROOT/'outputs/stage2/maps/endpoint-audit.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    raw=next(r for r in audit if not r['use_destination_id'])
    d=initial['route']['destination'];destination=Destination(d['id'],d['name'],Point(**d['point']),d['address'])
    actual_route=parse_amap_walking(raw['response'],destination)
    class RecordedProvider:
        def search(self,query,location):return [destination]
        def walking(self,origin,chosen):return actual_route
    nav=Navigation(RecordedProvider());tasks=Tasks();events=[]
    command='带我去江夏大道长岛公交站，到站后帮我等903路公交车'
    intent=parse_intent(command)
    events.append(dict(simulated_s=0,input_type='scripted_text',input=command,intent=intent))
    candidates=nav.search(intent['query'],arrival_wait=intent['arrival_wait'])
    events.append(dict(simulated_s=0,**candidates))
    events.append(dict(simulated_s=2,input_type='scripted_confirmation',input='选择第一个'))
    events.append(dict(simulated_s=2,**nav.confirm(1,Point(**initial['origin']['point']))))
    tracker=nav.tracker;records=[]
    for along in np.linspace(0,tracker.length,math.ceil(tracker.length/2)+1):
        seg=next((s for s in tracker.segments if s[2]+s[3]>=along),tracker.segments[-1])
        a,b,start,length,_=seg;t=min(1.,max(0.,(along-start)/length))
        point=Point(a.longitude+(b.longitude-a.longitude)*t,a.latitude+(b.latitude-a.latitude)*t)
        sim_s=3+along/1.2
        fix=Location(point,sim_s,2.,bearing(a,b),3.)
        event=nav.update(fix,sim_s);events.append(dict(simulated_s=sim_s,**event))
        if event.get('arrival_wait'):
            kind,text=tasks.command(event['arrival_wait'],sim_s)
            events.append(dict(simulated_s=sim_s,type=kind,text=text,wait=dict(tasks.wait)))
        records.append(dict(fix=asdict(fix),event=event,wait=dict(tasks.wait) if tasks.wait else None))
    assert records[-1]['event']['status']=='arrived'
    assert tasks.wait and tasks.wait['target']=='903'
    (out/'events.jsonl').write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in events),encoding='utf-8')
    (out/'route.json').write_text(json.dumps(asdict(actual_route),ensure_ascii=False,indent=2),encoding='utf-8')
    fonts={s:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',s) for s in [19,22,25,29,35,43]}
    points=[p for s in actual_route.steps for p in s.points];points+=[destination.point]
    lon0=min(p.longitude for p in points);lat0=min(p.latitude for p in points)
    factor=math.cos(math.radians(lat0));coords=[((p.longitude-lon0)*factor,p.latitude-lat0) for p in points]
    maxx=max(x for x,y in coords);maxy=max(y for x,y in coords);scale=min(890/maxx,525/maxy)
    def project(p):return (round(115+(p.longitude-lon0)*factor*scale),round(815-(p.latitude-lat0)*scale))
    def wrap(draw,text,xy,width,font,fill='#293d57'):
        x,y=xy;line=''
        for c in text:
            if draw.textlength(line+c,font=font)>width:
                draw.text((x,y),line,font=font,fill=fill);y+=font.size+10;line=''
            line+=c
        draw.text((x,y),line,font=font,fill=fill)
        return y+font.size+10
    first=actual_route.steps[0].points[0];last=actual_route.steps[-1].points[-1]
    poly=[project(p) for p in points[:-1]]
    def render(record,phase):
        img=Image.new('RGB',(1600,1000),'#f3f6fb');draw=ImageDraw.Draw(img)
        draw.text((45,28),'PRTS Core · 目的地确认 → 路线进度 → 到站等待',font=fonts[35],fill='#14263e')
        draw.text((45,92),'真实高德路线 + 仿真位置；无现场 GPS、街景或相机 AR 叠加',font=fonts[25],fill='#914514')
        draw.rounded_rectangle((40,170,1055,895),22,fill='white',outline='#c8d5e6',width=2)
        draw.rounded_rectangle((1080,170,1560,895),22,fill='white',outline='#c8d5e6',width=2)
        draw.text((72,191),'路线几何示意（GCJ-02）',font=fonts[25],fill='#243e60')
        draw.line(poly,fill='#9db4cc',width=9,joint='curve')
        for p,label,color in [(first,'武汉长岛','#347aa6'),(destination.point,'江夏大道长岛公交站','#268367')]:
            x,y=project(p);draw.ellipse((x-9,y-9,x+9,y+9),fill=color)
            tx=max(68,min(700,x-40));ty=max(235,min(849,y-36))
            draw.text((tx,ty),label,font=fonts[22],fill=color)
        if record:
            p=Point(**record['fix']['point']);x,y=project(p)
            draw.ellipse((x-13,y-13,x+13,y+13),fill='#e39426',outline='#73400a',width=2)
            e=record['event'];heading=record['fix']['camera_heading_deg']*math.pi/180
            draw.line((x,y,x+30*math.sin(heading),y-30*math.cos(heading)),fill='#73400a',width=4)
            draw.text((1110,208),'导航模块输出',font=fonts[29],fill='#14263e')
            y=wrap(draw,e['text'],(1110,280),410,fonts[29])
            draw.text((1110,y+28),f"步骤 {e['step_index']+1}/{len(actual_route.steps)}",font=fonts[25],fill='#425976')
            draw.text((1110,y+74),f"路线剩余 {e['remaining_m']:.0f} 米",font=fonts[25],fill='#425976')
            draw.text((1110,y+120),f"横向偏差 {e['cross_track_m']:.1f} 米",font=fonts[25],fill='#425976')
            if record['wait']:
                wrap(draw,'导航已结束，开始等待 903 路公交。此演示没有提供公交到站观测。',(1110,600),410,fonts[25],fill='#20775d')
            draw.text((72,933),f"仿真时间 {record['fix']['timestamp_s']:.1f} 秒 · 轨迹约 25 倍播放 · 非实测行走速度",font=fonts[22],fill='#596f8b')
        else:
            draw.text((1110,208),phase,font=fonts[29],fill='#14263e')
            y=wrap(draw,command,(1110,275),410,fonts[25])
            y=wrap(draw,'候选 1：'+destination.name+'，'+destination.address,(1110,y+30),410,fonts[25])
            wrap(draw,'脚本确认：选择第一个。确认后才开始规划和到达任务。',(1110,y+30),410,fonts[25],fill='#20775d')
            draw.text((72,933),'文字命令与候选选择由脚本提供；用于验证任务状态转换',font=fonts[22],fill='#596f8b')
        return img
    fps=15;writer=cv2.VideoWriter(str(out/'route-raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1600,1000))
    intro=render(None,'目的地确认');intro.save(out/'confirmation.png')
    for _ in range(fps*4):writer.write(cv2.cvtColor(np.asarray(intro),cv2.COLOR_RGB2BGR))
    for i,record in enumerate(records):
        img=render(record,'')
        writer.write(cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR))
        if i in (len(records)//3,len(records)-1):img.save(out/('arrival.png' if i==len(records)-1 else 'following.png'))
    for _ in range(fps*4):writer.write(cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR))
    writer.release()
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(out/'route-raw.mp4'),'-c:v','libx264','-crf','19','-pix_fmt','yuv420p',str(out/'route-demo.mp4')],check=True)
    report=dict(provenance='Actual AMap coordinate-only response, scripted command/confirmation, synthetic location following route at 1.2 m/s. No real GPS, camera, audio or street imagery.',
        source_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),route_length_m=tracker.length,steps=len(actual_route.steps),
        endpoint_to_poi_m=raw['endpoint_to_poi_m'],simulated_positions=len(records),arrival_wait=tasks.wait,
        video_fps=fps,approximate_trajectory_playback_speed=25,real_world_ar_registered=False)
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()

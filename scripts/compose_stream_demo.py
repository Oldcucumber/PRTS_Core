"""Compose recorded model output and runtime-generated speech on their measured clocks.

Playback is reconstructed, not a physical speaker/microphone measurement. The
source and this distinction remain visible throughout the resulting video.
"""
import argparse
import heapq
import html
import json
import math
import re
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def playback_schedule(events,entries,durations,start):
    """One speaker: priority, replacement, cancellation and expiry at availability."""
    requests={e['sequence']:e for e in events if e['type']=='speech_request'}
    nodes=[];serial=0
    def push(at,kind,data):
        nonlocal serial
        heapq.heappush(nodes,(at,serial,kind,data));serial+=1
    for event in events:
        if event['type'] in ('speech_request','speech_cancel'):push(event['emitted_s'],'control',event)
    for entry in entries:
        if entry['status']=='ready' and entry['sequence'] in requests:
            push(entry['audio_ready_s'],'ready',entry)
    latest={};waiting=[];active=None;clips=[]
    def stop(at):
        nonlocal active
        if active is not None:
            active['end_s']=min(active['end_s'],max(active['start_s'],at-start));active=None
    while nodes:
        at,_,kind,data=heapq.heappop(nodes)
        if kind=='control':
            if data['type']=='speech_cancel':
                latest.clear();waiting.clear();stop(at)
            else:
                group=data['replace_group'];latest[group]=data['sequence']
                waiting=[e for e in waiting if e['replace_group']!=group]
                if active and (active['replace_group']==group or data['priority']>active['priority']):stop(at)
        elif kind=='ready':
            request=requests[data['sequence']]
            if latest.get(request['replace_group'])==request['sequence'] and request['expires_s']>at:
                waiting.append(dict(data,priority=request['priority']))
                if active and request['priority']>active['priority']:stop(at)
        elif kind=='end' and active and active['sequence']==data:stop(at)
        waiting=[e for e in waiting if e['expires_s']>at and latest.get(e['replace_group'])==e['sequence']]
        if active is None and waiting:
            waiting.sort(key=lambda e:(-e['priority'],e['sequence']))
            e=waiting.pop(0)
            end=min(at+durations[e['file']],e['expires_s'])
            active=dict(e,start_s=at-start,end_s=end-start)
            clips.append(active);push(end,'end',e['sequence'])
    return [c for c in clips if c['end_s']>c['start_s']]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import cv2
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from scipy.signal import resample_poly
    import soundfile as sf
    report=json.loads((args.run/'summary.json').read_text(encoding='utf-8'))
    events=[json.loads(s) for s in (args.run/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    speech=json.loads((args.run/'speech/manifest.json').read_text(encoding='utf-8'))
    t0=report['started_monotonic_s'];case=report['scenario'];rate=24000
    source=Path(case['source']);source=source if source.is_absolute() else ROOT/source
    queue_demo=case.get('audit_kind')=='queue'
    brain_demo=case.get('audit_kind')=='brain_montage'
    queue_credit=None
    if queue_demo or brain_demo:
        queue_credit=json.loads((source.parent/'sources.json').read_text(encoding='utf-8'))
        (args.output/'sources.json').write_text(json.dumps(queue_credit,ensure_ascii=False,indent=2),encoding='utf-8')
    credit=None
    for metadata in (ROOT/'outputs/stage2/bus-generalization/data').glob('*sources*.json'):
        for item in json.loads(metadata.read_text(encoding='utf-8')):
            if (ROOT/item['local_file']).resolve()!=source.resolve():continue
            info=item['source_metadata'];ext=info['extmetadata']
            credit=dict(author=html.unescape(re.sub('<[^>]+>','',ext['Artist']['value'])),
                        license=ext['LicenseShortName']['value'],license_url=ext['LicenseUrl']['value'],
                        source=info['descriptionurl'],modification='PRTS inference overlay and reconstructed runtime speech')
    waves={};durations={}
    def waveform(path):
        samples,sr=sf.read(path,dtype='float32',always_2d=True);samples=samples.mean(1)
        if sr!=rate:
            g=math.gcd(sr,rate);samples=resample_poly(samples,rate//g,sr//g)
        return samples
    for e in speech['entries']:
        if e.get('status')=='ready':
            waves[e['file']]=waveform(args.run/'speech'/e['file']);durations[e['file']]=len(waves[e['file']])/rate
    schedule=playback_schedule(events,speech['entries'],durations,t0)
    cap=cv2.VideoCapture(str(args.run/'annotated.mp4'),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
    if not cap.isOpened():raise RuntimeError('Missing annotated recording')
    queue_cap=cv2.VideoCapture(str(source),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]) if queue_demo else None
    fps=cap.get(cv2.CAP_PROP_FPS);input_duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
    last_result=max([0.]+[e['emitted_s']-t0+2 for e in events if e['type'] in ('brain_observation','answer','target_observed','request_error')])
    duration=max([input_duration,last_result]+[e['end_s'] for e in schedule])+1
    mixed=np.zeros(round(duration*rate),np.float32)
    def add(at,wave,gain=1.):
        begin=round(at*rate);skip=max(0,-begin);begin=max(0,begin)
        n=min(len(wave)-skip,len(mixed)-begin)
        if n>0:mixed[begin:begin+n]+=wave[skip:skip+n]*gain
    for entry in case.get('user_audio',[]):
        path=Path(entry['file']);path=path if path.is_absolute() else ROOT/path
        add(entry['at'],waveform(path),.8)
    for e in schedule:add(e['start_s'],waves[e['file']][:round((e['end_s']-e['start_s'])*rate)],.9)
    for e in events:
        if e['type']=='sound_cue':
            n=int(.16*rate);time=np.arange(n)/rate
            tone=np.sin(2*np.pi*(880 if e['cue']=='target_found' else 550)*time)*np.sin(np.pi*np.arange(n)/n)*.12
            add(e['emitted_s']-t0,tone)
    np.clip(mixed,-.98,.98,out=mixed)
    sf.write(args.output/'playback.wav',mixed,rate)
    font_path='C:/Windows/Fonts/msyh.ttc'
    title=ImageFont.truetype(font_path,29);normal=ImageFont.truetype(font_path,24);small=ImageFont.truetype(font_path,19)
    def paragraph(draw,text,xy,font,fill,max_width):
        x,y=xy;line=''
        for c in text:
            if c=='\n' or (draw.textlength(line+c,font=font)>max_width and c not in '，。！？；：、'):
                draw.text((x,y),line,font=font,fill=fill);y+=font.size+9;line='' if c=='\n' else c
            else:line+=c
        if line:draw.text((x,y),line,font=font,fill=fill);y+=font.size+9
        return y
    writer=cv2.VideoWriter(str(args.output/'demo-raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1600,1000))
    event_index=0;history=[];last=None;screens=set();latest_evidence=None
    for index in range(math.ceil(duration*fps)):
        at=index/fps
        if at<input_duration:
            ok,frame=cap.read()
            if ok:last=frame
        while event_index<len(events) and events[event_index]['emitted_s']-t0<=at:
            e=events[event_index];event_index+=1
            if e['type']=='target_evidence':latest_evidence=e
            if e['type'] in ('transcript','answer','wait_started','wait_cancelled','target_candidate','target_observed','request_error','brain_observation'):
                history.append(e)
        canvas=Image.new('RGB',(1600,1000),'#eef2f7');draw=ImageDraw.Draw(canvas)
        draw.text((30,20),'PRTS Core · 连续输入与输出时序',font=title,fill='#15243c')
        input_label='真实照片时序拼接；合成语音 + 标明的文字命令' if queue_demo else '录制相机输入 1×'
        if brain_demo:input_label='真实照片与胸前录像拼接；脚本命令 + 合成用户语音'
        draw.text((30,64),f'经过 {at:.2f} 秒  |  '+(input_label if at<input_duration else '输入片段结束，等待后台输出'),font=small,fill='#50627b')
        if queue_cap:
            queue_cap.set(cv2.CAP_PROP_POS_MSEC,min(at,input_duration-0.5)*1000)
            ok,original=queue_cap.read()
            if ok:
                last=original
                def segment(t):return next((i for i,s in enumerate(case['image_segments']) if s['start']<=t<s['end']),None)
                if latest_evidence and segment(at)==segment(latest_evidence['observation_s']-t0):
                    colors={'called':(50,195,20),'counter':(20,100,255),'waiting':(0,190,255),'unconfirmed':(210,60,200)}
                    for item in latest_evidence['ocr']:
                        role=item.get('queue_role')
                        if role in colors:cv2.polylines(last,[np.round(item['box']).astype(np.int32)],True,colors[role],3)
        if last is not None:
            im=Image.fromarray(cv2.cvtColor(last,cv2.COLOR_BGR2RGB));im.thumbnail((920,800))
            canvas.paste(im,(30+(920-im.width)//2,110+(800-im.height)//2))
        y=110
        for e in history[-3:]:
            labels={'transcript':'输入识别','answer':'答复生成','wait_started':'等待开始','wait_cancelled':'取消等待','target_candidate':'候选未确认','target_observed':'目标确认','request_error':'处理错误','brain_observation':'多模态大脑观察'}
            draw.text((990,y),f'{labels[e["type"]]} · {e["emitted_s"]-t0:.2f}s',font=small,fill='#426180');y+=30
            text=e.get('text','')
            if e['type']=='brain_observation':
                text=e['observation']+f"\n决定 {e['decision']} · 生成时证据年龄 {e['age_s']:.1f}s"
            if len(text)>140:text=text[:140]+'…（完整结果见事件日志）'
            y=paragraph(draw,text,(990,y),normal,'#15243c',570)+22
        playing=next((e for e in schedule if e['start_s']<=at<e['end_s']),None)
        if playing:
            y=max(700,y)
            paragraph(draw,'合成播放：'+playing['text'],(990,y),small,'#007c6c',570)
        draw.text((30,914),'实测推理及运行时 TTS；声音按生成可用时刻重建播放，非现场扬声器录音。',font=small,fill='#50627b')
        mask_note='旧应用 UI 按场景配置遮挡。' if case.get('ignore_regions') else ''
        draw.text((30,944),mask_note+'路径与文字仅对应各自证据帧；此录像不是 Apple 真机结果。',font=small,fill='#50627b')
        if credit:
            draw.text((30,973),f"素材：{credit['author']} / Wikimedia Commons / {credit['license']}；已添加推理覆盖层，详见随附来源清单。",font=small,fill='#50627b')
        if queue_credit and not brain_demo:
            draw.text((30,973),'绿色：叫到号码；橙色：柜台；紫色：未确认。素材：Vorsea / Elite S Moramels，CC BY-SA 4.0 / 3.0。',font=small,fill='#50627b')
        if brain_demo:
            segment=next((s for s in case['image_segments'] if s['start']<=at<s['end']),None)
            note=segment['label'] if segment else '输入结束'
            draw.text((30,973),note+'；素材署名/许可见 sources.json；这是场景拼接，非同一次实地行程。',font=small,fill='#50627b')
        view=cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR);writer.write(view)
        bucket=int(at)//5
        if bucket not in screens:
            screens.add(bucket);canvas.save(args.output/f'frame-{int(at):03}.png')
    cap.release();writer.release()
    if queue_cap:queue_cap.release()
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(args.output/'demo-raw.mp4'),'-i',str(args.output/'playback.wav'),
        '-c:v','libx264','-crf','21','-pix_fmt','yuv420p','-c:a','aac','-b:a','128k','-shortest','-movflags','+faststart',str(args.output/'demo.mp4')],check=True)
    (args.output/'timeline.json').write_text(json.dumps(dict(source_run=str(args.run),duration_s=duration,
        input_duration_s=input_duration,playback=schedule,media_credit=credit,queue_media_credits=queue_credit,
        provenance='Actual recorded model/TTS availability with reconstructed single-speaker playback; not physical playback measurement.'),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(args.output/'demo.mp4'),speech_clips=len(schedule),duration_s=duration)))


if __name__=='__main__':main()

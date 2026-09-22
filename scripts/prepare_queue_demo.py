"""Real photos arranged as a declared montage, with synthetic spoken commands."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import pyttsx3

def main():
    output=ROOT/'outputs/stage2/streaming/queue-montage-input-v1'
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    files=[ROOT/'outputs/stage2/queue-holdout/data/commons-bank-queue.jpg',
           ROOT/'outputs/stage2/queue-holdout/data/commons-hk-foodcourt.jpg',
           ROOT/'outputs/stage2/queue-holdout/data-v2/masmitra.jpg']
    segments=[dict(start=0,end=12,file=str(files[0])),dict(start=12,end=32,file=str(files[1])),
              dict(start=32,end=48,file=str(files[2]))]
    frames=[]
    for file in files:
        frame=cv2.imread(str(file));h,w=frame.shape[:2];scale=min(1280/w,960/h)
        resized=cv2.resize(frame,(round(w*scale),round(h*scale)))
        canvas=np.zeros((960,1280,3),np.uint8);rh,rw=resized.shape[:2]
        canvas[(960-rh)//2:(960-rh)//2+rh,(1280-rw)//2:(1280-rw)//2+rw]=resized;frames.append(canvas)
    writer=cv2.VideoWriter(str(output/'montage-raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),2,(1280,960))
    for i in range(96):writer.write(frames[0 if i/2<12 else 1 if i/2<32 else 2])
    writer.release()
    movie=output/'montage.mp4'
    subprocess.run(['ffmpeg','-v','error','-i',str(output/'montage-raw.mp4'),'-c:v','libx264','-crf','16',
                    '-pix_fmt','yuv420p',str(movie)],check=True)
    engine=pyttsx3.init();voice=next(v.id for v in engine.getProperty('voices') if 'HUIHUI' in v.id.upper())
    engine.setProperty('voice',voice);engine.setProperty('rate',165)
    phrases=[(0,'wait3','帮我等三号，叫到时提醒我。'),(8,'wait215','改为等二百一十五号，叫到时提醒我。'),
             (23,'wait494','改为等四百九十四号，叫到时提醒我。'),(29,'cancel','取消等待。')]
    for at,name,text in phrases:engine.save_to_file(text,str(output/(name+'-raw.wav')))
    engine.runAndWait();audio=[]
    for at,name,text in phrases:
        file=output/(name+'.wav')
        subprocess.run(['ffmpeg','-v','error','-i',str(output/(name+'-raw.wav')),'-ac','1','-ar','16000',str(file)],check=True)
        audio.append(dict(at=at,file=str(file.relative_to(ROOT)),role='user',text=text,synthetic=True,
                          sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    case=dict(source=str(movie.relative_to(ROOT)),start=0,end=48,navigate=False,audit_kind='queue',
              user_audio=audio,text_commands=[dict(at=35,text='我的号码是 B003，叫到时提醒我'),dict(at=43,text='取消等待')],
              image_segments=segments,synthetic_input_audio=True,
              provenance='Three actual attributed Commons photos scaled/padded and held as a 48-second montage; not a real video of a changing queue. Microsoft Huihui synthetic user PCM, plus declared typed B003/cancel commands. No real-world ASR or queue-transition accuracy claim.')
    scenario=output/'scenario.json';scenario.write_text(json.dumps(case,ensure_ascii=False,indent=2),encoding='utf-8')
    sources=[]
    for metadata in ['data/sources.json','data-v2/sources.json']:
        for source in json.loads((ROOT/'outputs/stage2/queue-holdout'/metadata).read_text(encoding='utf-8')):
            if (ROOT/source['source']).resolve() in files:sources.append(source)
    (output/'sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'preparation_snapshot.py').write_bytes(Path(__file__).read_bytes())
    print(scenario)

if __name__=='__main__':main()

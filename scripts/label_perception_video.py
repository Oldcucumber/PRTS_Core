"""Make comparison status and the measured playback ratio readable, without rerunning models."""
import json,subprocess
from pathlib import Path
import cv2

def main():
    p=Path('outputs/stage2/perception/continuous-turn');rows=[json.loads(s) for s in (p/'frames.jsonl').read_text(encoding='utf-8').splitlines()]
    manifest=json.loads((p/'manifest.json').read_text());owners=[];written=0
    for i,r in enumerate(rows):
        count=max(1,round(r['processing_elapsed_s']*15)-written);owners.extend([i]*count);written+=count
    def paint(frame,index,source):
        r=rows[index]
        for x,key,label in [(0,'v1','V1 B4'),(640,'new','NEW M2')]:
            cv2.rectangle(frame,(x,360),(x+640,382),(10,12,15),-1)
            cv2.putText(frame,f'{label} | {r[key]["status"]} / {r[key]["direction"]}',(x+6,376),0,.5,(255,255,255),1)
        cv2.rectangle(frame,(0,720),(1280,744),(0,0,0),-1)
        rate=f'Source 1x; processing speed-up {manifest["source_video_processing_acceleration"]:.2f}x' if source else 'Measured pipeline 1x; source slowed'
        cv2.putText(frame,f'{rate} | source {r["source_time_s"]:.2f}s | processing {r["processing_elapsed_s"]:.2f}s',(8,740),0,.52,(255,255,255),1)
        return frame
    for name in ['source-time','processing-time']:
        cap=cv2.VideoCapture(str(p/(name+'.raw.mp4')));writer=cv2.VideoWriter(str(p/(name+'.labelled.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),15,(1280,768));index=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            source=name=='source-time';owner=index if source else owners[index];view=paint(frame,owner,source);writer.write(view)
            if source and owner in (0,30,60,90,120,150,179):cv2.imwrite(str(p/f'frame-{owner:06}.jpg'),view)
            index+=1
        cap.release();writer.release()
        subprocess.run(['D:/ffmpeg/bin/ffmpeg.exe','-y','-v','error','-i',str(p/(name+'.labelled.mp4')),
             '-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(p/(name+'.mp4'))],check=True)
        print(name,index,flush=True)

if __name__=='__main__':main()

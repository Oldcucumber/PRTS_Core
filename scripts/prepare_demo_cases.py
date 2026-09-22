"""Create local replay manifests; private media is referenced, never copied to Git."""
import argparse,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--private-dir',type=Path);p.add_argument('--skip-synthetic',action='store_true');args=p.parse_args()
    out=ROOT/'outputs/cases';out.mkdir(parents=True,exist_ok=True)
    def save(name,obj):(out/(name+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    sessions={'sidewalk':'0xCqEk5hjEvrygxu26MZkieSv45D_gaJ',
              'hydrant':'0zEhKDk1j7KSuUQYR_rmCLmlbb5FCYG6',
              'crowd':'1sftEbnzzIfYBdODrjDO9TbhcnsyKjAv',
              'labelled':'28tdwxgz-zPU06lpeDY3OPWTFxZyBv2c',
              'scaffold':'-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG',
              'holdout':'30S_d-kuvDkznn3Rhea0G3FoBQc5rXoA'}
    for name,sid in sessions.items():
        save(name,dict(source=f'outputs/data/sanpo/{sid}/manifest.json',navigate=True,interval=.25))
    if args.private_dir:
        cases=[('road','VID_20260422_131337_2 (1).mp4',5.1,7.0,6.9,7.0),
               ('notice','VID_20260422_131152 (1).mp4',3.2,5.0,5.0,5.1),
               ('bus','VID_20260422_132609 (1).mp4',3.0,8.9,9.0,33.5),
               ('crossing','VID_20260422_130910 (1).mp4',5.0,7.0,7.0,7.1)]
        for name,filename,a,b,start,end in cases:
            source=args.private_dir/filename;audio=out/(name+'_user.wav')
            subprocess.run(['ffmpeg','-v','error','-y','-ss',str(a),'-t',str(b-a),'-i',str(source),
                            '-vn','-ac','1','-ar','16000',str(audio)],check=True)
            save(name,dict(source=str(source),start=start,end=end,interval=.5,
                           roi=[0,.065,1,.87],ignore_regions=[[.73,0,1,.105],[.20,.95,.8,1]],
                           commands=[dict(at=start,audio=str(audio))]))
    # Offline SAPI synthesised inputs: integration fixtures, not real-world ASR evidence.
    if args.skip_synthetic:return
    import pyttsx3
    voice=pyttsx3.init()
    zh=next((v.id for v in voice.getProperty('voices') if 'HUIHUI' in v.id.upper()),None)
    if not zh:raise RuntimeError('Chinese offline SAPI voice unavailable; supply WAV command recordings instead')
    voice.setProperty('voice',zh);voice.setProperty('rate',160)
    phrases={'navigate':'请开始导航，帮我沿着人行道往前走。',
             'stop':'先停下来，不要继续给我指路了。',
             'scene':'看看我前面有什么东西挡着路？',
             'wait_number':'请帮我等待A108号，叫到我时提醒我。',
             'call_wrong':'请A180号到三号窗口办理业务。',
             'call_correct':'请A108号到三号窗口办理业务。',
             'cancel':'取消刚才的等待。'}
    for name,text in phrases.items():voice.save_to_file(text,str(out/(name+'.wav')))
    voice.runAndWait()
    (out/'synthetic_audio.json').write_text(json.dumps(dict(source='offline Microsoft Huihui TTS',synthetic=True,phrases=phrases),ensure_ascii=False,indent=2),encoding='utf-8')
    sid=sessions['sidewalk']
    save('voice_navigation',dict(source=f'outputs/data/sanpo/{sid}/manifest.json',interval=.25,
         commands=[dict(at=0,audio=str(out/'navigate.wav')),dict(at=2,audio=str(out/'scene.wav')),
                   dict(at=4,audio=str(out/'stop.wav'))]))
    from PIL import Image,ImageDraw,ImageFont
    font_path=next((p for p in [Path('C:/Windows/Fonts/msyh.ttc'),Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')] if p.exists()),None)
    if font_path is None:raise RuntimeError('A Chinese font is needed for synthetic queue fixtures')
    font=ImageFont.truetype(str(font_path),70)
    for name,number in [('wrong','A180'),('correct','A108')]:
        image=Image.new('RGB',(1280,720),(30,35,45));draw=ImageDraw.Draw(image)
        draw.text((100,90),'当前叫号',font=font,fill='white');draw.text((100,270),f'{number}号  请到3号窗口',font=font,fill=(90,240,150))
        draw.text((100,550),'合成测试画面',font=font,fill=(160,160,160));image.save(out/(name+'.png'))
    save('numbers_manifest',dict(synthetic=True,frames=[dict(file=('correct.png' if i>=10 else 'wrong.png'),time_s=i) for i in range(16)]))
    save('numbers',dict(source='outputs/cases/numbers_manifest.json',interval=1,synthetic=True,
        commands=[dict(at=0,audio='outputs/cases/wait_number.wav')],
        ambient=[dict(at=5,start=1,audio='outputs/cases/call_wrong.wav'),dict(at=9,start=6,audio='outputs/cases/call_correct.wav'),dict(at=14,start=11,audio='outputs/cases/call_correct.wav')]))
    save('numbers_visual',dict(source='outputs/cases/numbers_manifest.json',interval=1,synthetic=True,
        commands=[dict(at=0,audio='outputs/cases/wait_number.wav'),dict(at=13,audio='outputs/cases/cancel.wav')]))
    print(out)


if __name__=='__main__':main()

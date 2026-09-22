"""Download independently sourced real queue/advertisement photographs with credits."""
import hashlib
import json
from pathlib import Path
import requests
from PIL import Image,ImageDraw,ImageFont,ImageOps

ROOT=Path(__file__).resolve().parents[1]
FILES={
    'jtrust':'Display LCD TV di J Trust Bank.jpg',
    'masmitra':'Percobaan LCD TV di RS Masmitra Jati Makmur.jpg',
    'japan-led':'電光掲示板 (2363344795).jpg',
    'pharmacy':'Waiting number at a pharmacy, Oude Pekela (2019) 01.jpg',
    'burrito-ad':'Now Serving Breakfast Burrito.jpg',
}

def main():
    output=ROOT/'outputs/stage2/queue-holdout/data-v2'
    output.mkdir(parents=True,exist_ok=True)
    session=requests.Session();session.headers['User-Agent']='PRTSCoreResearch/0.2 (local vision evaluation; Commons attributed)'
    sources=[]
    for identity,title in FILES.items():
        response=session.get('https://commons.wikimedia.org/w/api.php',params=dict(action='query',format='json',
            titles='File:'+title,prop='imageinfo',iiprop='url|size|extmetadata'),timeout=45)
        response.raise_for_status();page=next(iter(response.json()['query']['pages'].values()));info=page['imageinfo'][0]
        path=output/(identity+'.jpg')
        if not path.exists():
            data=session.get(info['url'],timeout=45);data.raise_for_status();path.write_bytes(data.content)
        sources.append(dict(id=identity,source=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_metadata=info))
        print(identity,info['width'],info['height'],info['extmetadata']['LicenseShortName']['value'],flush=True)
    (output/'sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8')
    sheet=Image.new('RGB',(1500,1050),'#101822');draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',24)
    for i,source in enumerate(sources):
        x=(i%3)*500;y=(i//3)*525;image=Image.open(ROOT/source['source']).convert('RGB')
        preview=ImageOps.contain(image,(490,480));sheet.paste(preview,(x+(500-preview.width)//2,y+35))
        draw.text((x+10,y+2),source['id'],font=font,fill='white')
    sheet.save(output/'inspection.jpg')

if __name__=='__main__':main()

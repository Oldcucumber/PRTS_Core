"""Download a bounded SANPO-Real subset from its public CC-BY-4.0 bucket."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from pathlib import Path
import urllib.parse
import urllib.request

BASE = 'sanpo_dataset/v0/'
HOST = 'https://storage.googleapis.com/gresearch/'


def listing(prefix, delimiter=None, limit=1000):
    q = dict(prefix=prefix, maxResults=min(1000,limit))
    if delimiter:
        q['delimiter'] = delimiter
    result={'items':[],'prefixes':[]}
    while True:
        page=json.load(urllib.request.urlopen('https://storage.googleapis.com/storage/v1/b/gresearch/o?' + urllib.parse.urlencode(q), timeout=60))
        for key in ('items','prefixes'):result[key].extend(page.get(key,[]))
        if not page.get('nextPageToken') or len(result['prefixes'] if delimiter else result['items'])>=limit:break
        q['pageToken']=page['nextPageToken']
    return result


def fetch(name, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary=path.with_suffix(path.suffix+'.download')
        urllib.request.urlretrieve(HOST + urllib.parse.quote(name, safe='/'), temporary)
        temporary.replace(path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=Path('outputs/data/sanpo'))
    ap.add_argument('--discover', type=int, default=0)
    ap.add_argument('--session')
    ap.add_argument('--camera', default='camera_head')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=60)
    ap.add_argument('--stride', type=int, default=5)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.discover:
        sessions = listing(BASE+'sanpo-real/', '/', args.discover*2)['prefixes'][:args.discover]
        def inspect(prefix):
            sid = prefix.rstrip('/').split('/')[-1]
            desc = json.loads(fetch(prefix+'description.json', args.output/sid/'description.json').read_text())
            left = prefix+'camera_head/left/'
            folders = listing(left, '/').get('prefixes', [])
            frames = listing(left+'video_frames/', limit=1).get('items', [])
            if frames:
                fetch(frames[0]['name'], args.output/sid/'preview.png')
            return dict(session=sid, metadata=desc['session_video_metadata'], folders=folders)
        with ThreadPoolExecutor(max_workers=4) as pool:
            records = list(pool.map(inspect, sessions))
        (args.output/'discovery.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        import cv2
        import numpy as np
        tiles=[]
        for i,r in enumerate(records):
            frame=cv2.imread(str(args.output/r['session']/'preview.png'))
            if frame is None: continue
            frame=cv2.resize(frame,(384,216))
            cv2.rectangle(frame,(0,0),(384,24),(0,0,0),-1)
            cv2.putText(frame,f'{i}: {r["session"][:16]}',(5,17),0,.48,(255,255,255),1)
            tiles.append(frame)
        while len(tiles)%4: tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(args.output/'discovery.jpg'),np.concatenate([np.concatenate(tiles[i:i+4],axis=1) for i in range(0,len(tiles),4)]))
        print(json.dumps(records,indent=2))
    if args.session:
        prefix=BASE+'sanpo-real/'+args.session+'/'+args.camera+'/left/'
        records=listing(prefix+'video_frames/')['items']
        records=records[args.start:args.start+args.count*args.stride:args.stride]
        dest=args.output/args.session
        desc_name=BASE+'sanpo-real/'+args.session+'/description.json'
        fetch(desc_name,dest/'description.json')
        fetch(BASE+'labelmap.json',args.output/'labelmap.json')
        masks=listing(prefix+'segmentation_masks/').get('items',[])
        masks={Path(x['name']).name:x['name'] for x in masks}
        annotation_path=fetch(prefix+'frame_segmentation_annotation_type.json',dest/'annotation_type.json') if masks else None
        annotation=json.loads(annotation_path.read_text()) if annotation_path else {}
        manifest=[]
        def download(pair):
            i,item=pair;name=Path(item['name']).name
            fetch(item['name'],dest/'frames'/name)
            if name in masks: fetch(masks[name],dest/'labels'/name)
            return dict(file='frames/'+name, source=HOST+item['name'],
                        time_s=int(Path(name).stem)/15, label=('labels/'+name if name in masks else None),
                        annotation_type=annotation.get(str(int(Path(name).stem)),'UNLABELLED'),
                        image_sha256=hashlib.sha256((dest/'frames'/name).read_bytes()).hexdigest(),
                        label_sha256=hashlib.sha256((dest/'labels'/name).read_bytes()).hexdigest() if name in masks else None)
        with ThreadPoolExecutor(max_workers=4) as pool: manifest=list(pool.map(download,enumerate(records)))
        (dest/'manifest.json').write_text(json.dumps(dict(license='CC-BY-4.0',source='https://github.com/google-research-datasets/sanpo_dataset',camera=args.camera,frames=manifest),indent=2),encoding='utf-8')
        print(f'Downloaded {len(manifest)} frames to {dest}')


if __name__=='__main__': main()

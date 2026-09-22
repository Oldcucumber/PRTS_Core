"""Freeze/download SANPO chest clips and a separate labelled head-camera subset.

The chest clips have NO public segmentation truth. Head labels must never be
paired with chest images. Source PNGs and SHA-256 hashes are retained.
"""
import argparse,hashlib,json,sys,urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from sanpo_samples import BASE,HOST,fetch,listing

CASES=[
 ('dev-sidewalk','0xCqEk5hjEvrygxu26MZkieSv45D_gaJ','development'),
 ('dev-hydrant','0zEhKDk1j7KSuUQYR_rmCLmlbb5FCYG6','development'),
 ('dev-crowd','1sftEbnzzIfYBdODrjDO9TbhcnsyKjAv','development'),
 ('dev-turn','28tdwxgz-zPU06lpeDY3OPWTFxZyBv2c','development'),
 ('dev-scaffold','-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG','development'),
 ('dev-snow','30S_d-kuvDkznn3Rhea0G3FoBQc5rXoA','development'),
 ('test-park','6Zngp12ETEAAZnfliZdHRFJBJQU9V3ZU','heldout'),
 ('test-urban-a','6lm2fRHPl4AtOLKJkEXRJo9MQJVeCT_y','heldout'),
 ('test-urban-b','6uEJtqNVqHJA1tjeemumA-trgb9xnvt0','heldout'),
 ('test-park-b','7PoccorPXoVcJawbvbCZizt4yBHF2SeB','heldout'),
 ('test-rural','8bWymqLVx34nEZhZek77q8WqIEvRucxE','heldout'),
 ('test-jog','5wiNc10ZAN5qnULkU6whKbRb8OGFLxUl','heldout'),
]

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare(case,root,preview=False):
    name,sid,split=case;dest=root/name;prefix=BASE+'sanpo-real/'+sid+'/'
    fetch(prefix+'description.json',dest/'description.json')
    fetch(prefix+'camera_chest/fixed_camera_poses.csv',dest/'fixed_camera_poses.csv')
    items=listing(prefix+'camera_chest/left/video_frames/',limit=180)['items']
    selected=items[:1] if preview else items[:180] if name=='dev-turn' else items[:160:4]
    def image(item):
        fn=Path(item['name']).name;path=fetch(item['name'],dest/'chest'/fn)
        return {'file':'chest/'+fn,'time_s':int(Path(fn).stem)/15,'source':HOST+item['name'],
                'image_sha256':digest(path),'annotation_type':'UNLABELLED','label':None}
    with ThreadPoolExecutor(max_workers=4) as pool:frames=list(pool.map(image,selected))
    head=[]
    if not preview:
        lp=prefix+'camera_head/left/'
        masks=listing(lp+'segmentation_masks/',limit=300)['items']
        if masks:
            ann=fetch(lp+'frame_segmentation_annotation_type.json',dest/'head_annotation_type.json')
            ann=json.loads(ann.read_text())
            human=[x for x in masks if ann.get(str(int(Path(x['name']).stem)))=='HUMAN_ANNOTATED'][:8]
            for item in human:
                fn=Path(item['name']).name
                label=fetch(item['name'],dest/'head-labels'/fn)
                pic=fetch(lp+'video_frames/'+fn,dest/'head'/fn)
                head.append({'file':'head/'+fn,'time_s':int(Path(fn).stem)/15,'source':HOST+lp+'video_frames/'+fn,
                    'image_sha256':digest(pic),'label':'head-labels/'+fn,'label_source':HOST+item['name'],
                    'label_sha256':digest(label),'annotation_type':'HUMAN_ANNOTATED'})
    manifest={'case':name,'session':sid,'split':split,'camera':'camera_chest','source_fps':15,
              'frame_stride':1 if name=='dev-turn' else 4,'license':'CC-BY-4.0',
              'source':'https://github.com/google-research-datasets/sanpo_dataset','frames':frames,
              'human_labels_camera':'camera_head','human_label_frames':head,
              'selection':'Fixed by session metadata before Stage 2 candidate inference; development sessions seen in V1; heldout sessions excluded from V1 training.'}
    (dest/('preview-manifest.json' if preview else 'manifest.json')).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(name,len(frames),'chest,',len(head),'human-labelled head',flush=True)
    return {'case':name,'split':split,'session':sid,'manifest':name+'/manifest.json'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('outputs/stage2/perception/data'))
    ap.add_argument('--preview',action='store_true');ap.add_argument('--case');a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);fetch(BASE+'labelmap.json',a.output/'labelmap.json')
    cases=[c for c in CASES if not a.case or c[0]==a.case]
    records=[]
    for c in cases:records.append(prepare(c,a.output,a.preview))
    (a.output/('previews.json' if a.preview else 'dataset.json')).write_text(json.dumps(records,indent=2))

if __name__=='__main__':main()

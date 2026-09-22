"""Small Latin OCR alternatives on already-failed development crop views."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import rapidocr
from omegaconf import OmegaConf
from rapidocr.utils.typings import LangRec,ModelType,OCRVersion
from prts_core.tasks import identifiers


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--crops',type=Path,default=ROOT/'outputs/stage2/bus-generalization/crop-probe-v1')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--profile',choices=['english','v6-medium','v5-server'],default='english')
    ap.add_argument('--detector',choices=['small','medium'],default='small')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'results.json').exists():raise FileExistsError('Preserve the earlier OCR probe')
    source=Path(rapidocr.__file__).parent
    catalog=OmegaConf.to_container(OmegaConf.load(source/'default_models.yaml'))['onnxruntime']
    detector=source/'models/PP-OCRv6_det_small.onnx'
    if a.detector=='medium':
        entry=catalog['PP-OCRv6']['det']['multi_PP-OCRv6_det_medium']
        detector=ROOT/'models/ocr-candidates/PP-OCRv6_det_medium.onnx'
        detector.parent.mkdir(exist_ok=True)
        if not detector.exists():
            with urlopen(entry['model_dir'],timeout=60) as r:detector.write_bytes(r.read())
        if hashlib.sha256(detector.read_bytes()).hexdigest()!=entry['SHA256']:raise ValueError('Detector hash mismatch')
    frozen=json.loads((a.crops/'results.json').read_text(encoding='utf-8'));results=[]
    profiles=([('PP-OCRv4','en_PP-OCRv4_rec_mobile',LangRec.EN,ModelType.MOBILE),('PP-OCRv5','en_PP-OCRv5_rec_mobile',LangRec.EN,ModelType.MOBILE)] if a.profile=='english' else
              [('PP-OCRv6','multi_PP-OCRv6_rec_medium',LangRec.CH,ModelType.MEDIUM)] if a.profile=='v6-medium' else
              [('PP-OCRv5','ch_PP-OCRv5_rec_server',LangRec.CH,ModelType.SERVER)])
    for version,key,language,model_type in profiles:
        entry=catalog[version]['rec'][key]
        model=ROOT/'models/ocr-candidates'/(key+'.onnx');model.parent.mkdir(exist_ok=True)
        if not model.exists():
            with urlopen(entry['model_dir'],timeout=60) as r:model.write_bytes(r.read())
        sha=hashlib.sha256(model.read_bytes()).hexdigest()
        if sha!=entry['SHA256']:raise ValueError('Official OCR hash mismatch')
        engine=rapidocr.RapidOCR(params={
            'Det.model_path':str(detector),
            'Cls.model_path':str(source/'models/ch_ppocr_mobile_v2.0_cls_mobile.onnx'),
            'Rec.model_path':str(model),'Rec.lang_type':language,'Rec.model_type':model_type,'Rec.ocr_version':OCRVersion(version),
            'Global.max_side_len':960,'Det.limit_type':'max','Det.limit_side_len':960,'Global.use_cls':False,
            'Global.log_level':'warning','EngineConfig.onnxruntime.intra_op_num_threads':4,
            'EngineConfig.onnxruntime.inter_op_num_threads':1})
        rows=[]
        for case in frozen['rows']:
            file=a.crops/f'{case["case"]}-{case["view"]}.png'
            if hashlib.sha256(file.read_bytes()).hexdigest()!=case['image_sha256']:raise ValueError('Frozen crop changed')
            r=engine(cv2.imread(str(file)));entries=[] if r.txts is None else [dict(text=str(t),score=float(s),box=np.asarray(b).tolist()) for t,s,b in zip(r.txts,r.scores,r.boxes)]
            route=case['verification']['route_id']
            joint=bool(route and any(route in identifiers(e['text']) and e['score']>=.45 for e in entries))
            rows.append(dict(case=case['case'],view=case['view'],ocr=entries,vlm_route=route,confirmed_by_both=joint))
            print(json.dumps(dict(version=version,case=case['case'],view=case['view'],ocr=[e['text'] for e in entries],joint=joint),ensure_ascii=False),flush=True)
        results.append(dict(version=version,source=entry['model_dir'],sha256=sha,bytes=model.stat().st_size,rows=rows,
                            detector=str(detector),detector_sha256=hashlib.sha256(detector.read_bytes()).hexdigest()))
        del engine;gc.collect()
    (a.output/'results.json').write_text(json.dumps(dict(scope='Development OCR comparison, using previous target-independent VLM readings of hash-matched crops',results=results),ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()

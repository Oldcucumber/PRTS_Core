"""Compare actual C++ paths with the frozen Python geometry, not perception accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from prts_core.native_geometry import NativeGeometry
from prts_core.guidance import search_corridor,validate_path,path_heading


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--library',type=Path,default=ROOT/'outputs/native-toolchain/prts_geometry.dll')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--mask',type=Path,action='append',default=[])
    ap.add_argument('--mask-dir',type=Path)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);native=NativeGeometry(a.library)
    masks={};rng=np.random.default_rng(51327)
    for w,h in [(160,90),(320,180),(181,320)]:
        masks[f'free-{w}-{h}']=np.ones((h,w),bool)
        masks[f'closed-{w}-{h}']=np.zeros((h,w),bool)
        for turn in [-1,0,1]:
            raw=np.zeros((h,w),np.uint8)
            pts=np.array([[w//2,h-1],[w//2,int(h*.80)],[round(w*(.5+.26*turn)),int(h*.59)],
                          [round(w*(.5+.26*turn)),int(h*.30)]],np.int32)
            cv2.polylines(raw,[pts],False,1,round(w*.19))
            masks[f'bend-{w}-{h}-{turn}']=raw.astype(bool)
        for j in range(8):
            raw=np.ones((h,w),np.uint8)
            for k in range(12):
                x,y=rng.integers([0,round(.42*h)],[w,round(.9*h)])
                cv2.rectangle(raw,(int(x),int(y)),(int(x+w*.04),int(y+h*.05)),0,-1)
            masks[f'obstacles-{w}-{h}-{j}']=raw.astype(bool)
    for p in a.mask:
        masks[p.stem]=np.load(p).astype(bool)
    if a.mask_dir:
        for p in sorted(a.mask_dir.glob('*.npz')):
            with np.load(p) as data:masks[p.stem]=data['sidewalk_class'] & (data['sidewalk']>=.50)
    results=[]
    for name,mask in masks.items():
        previous=None
        for prefer in [None,-1,1]:
            reference,coverage=search_corridor(mask,prefer=prefer,prior=previous)
            actual,native_coverage=native.search(mask,prefer=prefer,prior=previous)
            validation=native.validate(actual,mask);expected=validate_path(actual,mask)
            angles=native.heading(actual,mask.shape) if actual else {}
            reference_angles=path_heading(actual,mask.shape) if actual else {}
            angle_error=max([abs(v-reference_angles[k]) for k,v in angles.items()] or [0])
            exact=reference==actual
            item=dict(mask=name,preference=prefer,points=len(actual),reference_points=len(reference),
                      exact_path=exact,coverage_error=abs(coverage-native_coverage),
                      validation_matches=all(v==expected[k] for k,v in validation.items()),angle_error=angle_error,
                      passed=exact and abs(coverage-native_coverage)<1e-12 and angle_error<1e-10 and all(v==expected[k] for k,v in validation.items()))
            results.append(item);previous=reference
            if not item['passed']:
                (a.output/f'{name}-{prefer}.json').write_text(json.dumps(dict(reference=reference,actual=actual),indent=2))
    np.savez_compressed(a.output/'input-masks.npz',**masks)
    broken=[]
    for name,points in [('hole',[(80,70),(80,20)]),('out_of_bounds',[(-2,40),(162,40)]),('empty',[]),('point',[(80,45)])]:
        mask=np.ones((90,160),bool);mask[45,80]=False
        expected=validate_path(points,mask);actual=native.validate(points,mask)
        broken.append(dict(case=name,passed=all(v==expected[k] for k,v in actual.items()),**actual))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'prts_core/guidance.py',
        ROOT/'native/portable/prts_geometry.cpp',ROOT/'native/portable/include/prts_geometry.h',a.library]}
    report=dict(passed=all(r['passed'] for r in results+broken),cases=len(results),polyline_checks=broken,
                failed=[r for r in results if not r['passed']],results=results,sha256=hashes,
                scope='Constructed geometry and optional exported masks; not ground-truth recognition or Apple execution.')
    (a.output/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],cases=len(results),failed=len(report['failed']))))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())

"""Fixed model-level migration gate on the existing 14-frame export audit set."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from prts_core.native_image import NativeImage
from prts_core.onnx_perception import ONNXPerception
from perception_metrics import label_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--provider', default='DmlExecutionProvider')
    ap.add_argument('--semantic', type=Path, default=Path('outputs/stage2/perception/onnx/mask2former-1024-fp16-v2'))
    ap.add_argument('--data', type=Path, default=Path('outputs/stage2/perception/data'))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result_file = args.output / 'results.json'
    if result_file.exists():
        raise FileExistsError('Preserve the earlier migration audit')
    log=(args.output/'runtime-stderr.log').open('wb');os.dup2(log.fileno(),2)
    gates = dict(mean_class_agreement_at_least=.98, minimum_walkable_iou=.95,
                 maximum_human_walkable_iou_drop=.01)
    frozen = []
    for path in sorted(args.data.glob('dev-*/manifest.json')):
        manifest = json.loads(path.read_text(encoding='utf-8'))
        records = [manifest['frames'][len(manifest['frames']) // 2], *manifest['human_label_frames']]
        for record in records:
            image = path.parent / record['file']
            frozen.append(dict(key=manifest['case'] + '-' + image.parent.name + '-' + image.stem,
                               image=str(image), label=str(path.parent / record['label']) if record.get('label') else None,
                               sha256=hashlib.sha256(image.read_bytes()).hexdigest()))
    (args.output / 'frozen-inputs-and-gates.json').write_text(
        json.dumps(dict(inputs=frozen, gates=gates), indent=2), encoding='utf-8')
    shutil.copyfile(__file__, args.output / 'validator.py.snapshot')
    library = ROOT / 'outputs/native-toolchain/prts_image.dll'
    native = NativeImage(library)
    model = ONNXPerception(args.semantic, low_memory=True, threads=4, providers=[args.provider])
    ih, iw = model.shape[2:]
    rows = []
    for item in frozen:
        bgr = cv2.imread(item['image'])
        reference = model(bgr, detect=False)
        tensor, crop = native.letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), iw, ih,
                                       model.meta['mean'], model.meta['std'])
        scores = model.seg.run(None, {'rgb': tensor})[0][0]
        gh, gw = reference['classes'].shape
        candidate = native.decode(scores, crop, iw, ih, gw, gh, model.sidewalk_ids)
        candidate['sidewalk_class'] = np.isin(candidate['classes'], model.sidewalk_ids)
        ref_walk = reference['sidewalk_class'] & (reference['sidewalk'] >= .5)
        walk = candidate['walkable']
        row = dict(key=item['key'], class_agreement=float((reference['classes'] == candidate['classes']).mean()),
                   walkable_iou=float((walk & ref_walk).sum() / max(1, (walk | ref_walk).sum())))
        if item['label']:
            row['reference_label_metrics'] = label_metrics(reference, item['label'])
            row['candidate_label_metrics'] = label_metrics(candidate, item['label'])
            row['human_walkable_iou_drop'] = row['reference_label_metrics']['walkable_iou'] - row['candidate_label_metrics']['walkable_iou']
        np.savez_compressed(args.output / (item['key'] + '.npz'),
                            reference_classes=reference['classes'], native_classes=candidate['classes'],
                            reference_walkable=ref_walk, native_walkable=walk)
        rows.append(row)
        print(json.dumps(row), flush=True)
    measured = dict(mean_class_agreement=float(np.mean([r['class_agreement'] for r in rows])),
                    minimum_walkable_iou=min(r['walkable_iou'] for r in rows),
                    maximum_human_walkable_iou_drop=max(r.get('human_walkable_iou_drop', 0) for r in rows))
    passed = (measured['mean_class_agreement'] >= gates['mean_class_agreement_at_least']
              and measured['minimum_walkable_iou'] >= gates['minimum_walkable_iou']
              and measured['maximum_human_walkable_iou_drop'] <= gates['maximum_human_walkable_iou_drop'])
    sources = [library, ROOT / 'native/portable/prts_image.cpp', args.semantic / 'semantic.onnx']
    result_file.write_text(json.dumps(dict(passed=passed, gates=gates, measured=measured, rows=rows,
        provider=model.seg.get_providers(), sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        scope='Pre/postprocessing migration on the previously used export audit set; not unseen field accuracy and not Apple execution.'), indent=2), encoding='utf-8')
    print(json.dumps(dict(passed=passed, measured=measured)), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())

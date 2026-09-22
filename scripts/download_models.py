"""Explicit online preparation. Runtime uses only these local artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import snapshot_download

MODELS = {
    'seg-mask2former': ('facebook/mask2former-swin-large-mapillary-vistas-semantic', '4772b6bf101d91f2534c106dc524d906aeb3c68a', 'mask2former-mapillary'),
    'brain': ('openbmb/MiniCPM-V-4.6', '36f34a661a4bd35d0dc2294cb044d2584646c7d3', 'minicpm-v-4.6'),
    'asr': ('Systran/faster-whisper-small', '536b0662742c02347bc0e980a01041f333bce120', 'whisper-small'),
    'asr-medium': ('Systran/faster-whisper-medium', '08e178d48790749d25932bbc082711ddcfdfbc4f', 'whisper-medium'),
    'seg': ('nvidia/segformer-b0-finetuned-cityscapes-1024-1024', '21b3847fae21ddee674abd31129307b6a1235bd9', 'segformer-cityscapes'),
    'seg-sidewalk': ('nickmuchi/segformer-b4-finetuned-segments-sidewalk', 'a8ca92dc8795137a2c54e00ada2c4dbcdfa79be0', 'segformer-sidewalk'),
    'seg-b5': ('nvidia/segformer-b5-finetuned-cityscapes-1024-1024', '2c6f153e4c23c229e2fa2b188eb250607e030cd8', 'segformer-cityscapes-b5'),
}
URL_MODELS={
    'detector': ('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt','yolo11n.pt'),
    'seg-yolo26': ('https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-sem.pt','yolo26l-sem.pt'),
    'seg-yolo26s': ('https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-sem.pt','yolo26s-sem.pt'),
    'seg-yolo26m': ('https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-sem.pt','yolo26m-sem.pt'),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('components', nargs='+', choices=[*MODELS, *URL_MODELS])
    p.add_argument('--root', type=Path, default=Path('models'))
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    for key in args.components:
        if key in URL_MODELS:
            url,filename=URL_MODELS[key]
            target = args.root / filename
            if not target.exists():
                temp = target.with_suffix('.download')
                urllib.request.urlretrieve(url, temp)
                temp.replace(target)
            manifest = {'source': url, 'files': {target.name: {'bytes': target.stat().st_size,
                        'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}}}
        else:
            repo, revision, folder = MODELS[key]
            print(f'Preparing {repo} at {revision}', flush=True)
            target = Path(snapshot_download(repo, revision=revision, local_dir=args.root / folder,
                ignore_patterns=['tf_model.h5','runs/*','training_args.bin']+(['pytorch_model.bin'] if key=='seg-mask2former' else [])))
            manifest = {'source': repo, 'revision': revision, 'files': {}}
            for f in sorted(target.iterdir()):
                if f.is_file():
                    h = hashlib.sha256()
                    with f.open('rb') as stream:
                        for block in iter(lambda: stream.read(8*1024*1024), b''):
                            h.update(block)
                    manifest['files'][f.name] = {'bytes': f.stat().st_size, 'sha256': h.hexdigest()}
        (args.root / f'{key}-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(f'{key}: ready', flush=True)


if __name__ == '__main__':
    main()

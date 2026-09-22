"""Fetch the pinned official MiniCPM-V 4.6 quantized weights for CPU/Metal trials."""
from pathlib import Path
from urllib.request import urlopen
import argparse
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
REVISION='afe9accb78d2995d214cd912920c9c92f4015faa'
HASHES={
    'MiniCPM-V-4_6-Q4_K_M.gguf':'6b0c74962c44bc6bf4b655b9b02c13eda9d5a0491543ae976d1ac18e4b7892e2',
    'MiniCPM-V-4_6-Q5_K_M.gguf':'513be695adfbf81ab0e6c18835eba887b74d13ed1ecb6931b85ac65ab5726e32',
    'MiniCPM-V-4_6-F16.gguf':'34754a0bc132e94de58d4ff50041b3481ebbad19bcde01977bdf32d5b0c9c547',
    'mmproj-model-f16.gguf':'ca931d861d0801d9003e50697cd764721a334107c0e0415a51168ee1938462de'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--language',choices=['Q5_K_M','Q4_K_M','F16'],default='Q5_K_M');args=ap.parse_args()
    out=ROOT/'models/minicpm-v-4.6-gguf';out.mkdir(parents=True,exist_ok=True)
    entries=[]
    for name in [f'MiniCPM-V-4_6-{args.language}.gguf','mmproj-model-f16.gguf']:
        url=f'https://huggingface.co/openbmb/MiniCPM-V-4.6-gguf/resolve/{REVISION}/{name}'
        p=out/name
        if not p.exists():
            partial=p.with_suffix('.partial')
            with urlopen(url,timeout=60) as r,partial.open('wb') as f:
                size=0;notice=0
                while chunk:=r.read(4*1024*1024):
                    f.write(chunk);size+=len(chunk)
                    if size-notice>=256*1024*1024:
                        print(json.dumps({'file':name,'downloaded_bytes':size}),flush=True);notice=size
            with partial.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
            if sha!=HASHES[name]:raise ValueError('Pinned model hash differs for '+name)
            partial.replace(p)
        with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        if sha!=HASHES[name]:raise ValueError('Pinned model hash differs for '+name)
        entries.append(dict(file=name,url=url,bytes=p.stat().st_size,sha256=sha))
        print(json.dumps({'file':name,'bytes':p.stat().st_size,'hash_verified':True}),flush=True)
    (out/f'manifest-{args.language}.json').write_text(json.dumps(dict(repository='openbmb/MiniCPM-V-4.6-gguf',revision=REVISION,
        license='Apache-2.0',files=entries,validation='Pinned hashes verified; platform/effect results are recorded separately in docs/RUNTIME_RESULTS.md'),indent=2),encoding='utf-8')


if __name__=='__main__':main()

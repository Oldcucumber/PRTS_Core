"""Fetch an optional pinned language candidate and verify its official LFS hash."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
ROOT=Path(__file__).resolve().parents[1]
REVISION='afe9accb78d2995d214cd912920c9c92f4015faa'
CANDIDATES={
    'Q5_K_M':(577802944,'513be695adfbf81ab0e6c18835eba887b74d13ed1ecb6931b85ac65ab5726e32'),
    'Q8_0':(811591616,'5cc8be0b5fb0c5bd3ad4e017000a9055a4e7dd09ca41a068dfb245e43d7422e4')
}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('candidate',choices=list(CANDIDATES));a=ap.parse_args()
    size,expected=CANDIDATES[a.candidate];name=f'MiniCPM-V-4_6-{a.candidate}.gguf'
    out=ROOT/'models/minicpm-v-4.6-gguf';file=out/name
    url=f'https://huggingface.co/openbmb/MiniCPM-V-4.6-gguf/resolve/{REVISION}/{name}'
    if not file.exists():
        partial=file.with_suffix('.partial')
        with urlopen(url,timeout=60) as src,partial.open('wb') as dst:
            total=0;last=0
            while chunk:=src.read(4*1024*1024):
                dst.write(chunk);total+=len(chunk)
                if total-last>128*1024*1024:print(json.dumps(dict(file=name,bytes=total)),flush=True);last=total
        if partial.stat().st_size!=size:raise ValueError('Unexpected file size')
        with partial.open('rb') as src:digest=hashlib.file_digest(src,'sha256').hexdigest()
        if digest!=expected:raise ValueError('Official hash mismatch')
        partial.replace(file)
    with file.open('rb') as src:digest=hashlib.file_digest(src,'sha256').hexdigest()
    if digest!=expected:raise ValueError('Existing model hash mismatch')
    (out/f'{a.candidate}-candidate-manifest.json').write_text(json.dumps(dict(file=name,bytes=size,sha256=digest,
        revision=REVISION,url=url,license='Apache-2.0',status='Downloaded for comparison, not adopted'),indent=2),encoding='utf-8')
    print(json.dumps(dict(file=name,sha256=digest,verified=True)))


if __name__=='__main__':main()

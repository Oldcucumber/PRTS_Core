"""Download the pinned official sherpa-onnx SenseVoice ONNX candidate."""
from pathlib import Path
from urllib.request import urlopen
import hashlib
import json
import tarfile

ROOT=Path(__file__).resolve().parents[1]
URL='https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09.tar.bz2'


def main():
    out=ROOT/'models/sensevoice-int8';out.mkdir(parents=True,exist_ok=True)
    archive=out/'source.tar.bz2'
    if not archive.exists():
        partial=archive.with_suffix('.partial')
        with urlopen(URL,timeout=45) as r,partial.open('wb') as f:
            while chunk:=r.read(1024*1024):f.write(chunk)
        partial.replace(archive)
    with tarfile.open(archive,'r:bz2') as tar:
        for entry in tar:
            name=Path(entry.name).name
            if entry.isfile() and name in ('model.int8.onnx','tokens.txt','README.md','LICENSE'):
                with tar.extractfile(entry) as source,(out/name).open('wb') as dest:
                    while chunk:=source.read(1024*1024):dest.write(chunk)
    files=[]
    for name in ('model.int8.onnx','tokens.txt'):
        p=out/name;digest=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
        files.append(dict(file=name,bytes=p.stat().st_size,sha256=digest))
    result=dict(source=URL,version='2025-09-09',files=files,validation='downloaded; inference validation pending')
    (out/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()

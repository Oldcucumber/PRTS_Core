"""Pinned alternative visual brain; hashes verified before a file is accepted."""
import hashlib
import argparse
import json
from pathlib import Path
import requests
from concurrent.futures import ThreadPoolExecutor,as_completed

ROOT=Path(__file__).resolve().parents[1]
REPO='unsloth/Qwen3.5-2B-GGUF'
REV='f6d5376be1edb4d416d56da11e5397a961aca8ae'
FILES=[('Qwen3.5-2B-Q5_K_M.gguf',1435238656,'1885b3a9195f8cc09da9a7a7a75afdc1e8d5cbf9fc4a499c3961dddea37098ac'),
       ('mmproj-F16.gguf',668227264,'7035e9cb8d7c6a9681d07eef9a364783e86ea4cd73faab2eabb4f43a101830c7')]

def ranged_download(url,path,size):
    chunk=32*1024*1024
    with path.open('wb') as f:f.truncate(size)
    def fetch(start):
        end=min(size,start+chunk)-1;count=0
        with requests.get(url+f'?download=true&range={start}',headers={'Range':f'bytes={start}-{end}'},
                          stream=True,timeout=(20,90)) as r:
            r.raise_for_status()
            if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':
                raise RuntimeError('Server did not honor the requested byte range')
            with path.open('r+b') as f:
                f.seek(start)
                for data in r.iter_content(1024*1024):f.write(data);count+=len(data)
        if count!=end-start+1:raise RuntimeError('Incomplete byte range')
        return count
    done=0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(fetch,start) for start in range(0,size,chunk)]):
            done+=future.result();print(f'{path.stem}: {done}/{size} bytes',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--size',choices=['2b','4b','vl4b','vl2b'],default='2b');args=ap.parse_args()
    repo,rev,files=REPO,REV,FILES
    if args.size=='4b':
        repo='unsloth/Qwen3.5-4B-GGUF';rev='e87f176479d0855a907a41277aca2f8ee7a09523'
        files=[('Qwen3.5-4B-Q4_K_M.gguf',2740937888,'00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4'),
               ('mmproj-F16.gguf',672423616,'cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864')]
    folder=ROOT/f'models/qwen3.5-{args.size}-gguf'
    upstream='Qwen/Qwen3.5-'+args.size.upper()
    if args.size=='vl4b':
        repo='Qwen/Qwen3-VL-4B-Instruct-GGUF';rev='1cd86afb9a95c410a6038ab3b40d8b578c892266'
        upstream='Qwen/Qwen3-VL-4B-Instruct';folder=ROOT/'models/qwen3-vl-4b-gguf'
        files=[('Qwen3VL-4B-Instruct-Q4_K_M.gguf',2497281664,'66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a'),
               ('mmproj-Qwen3VL-4B-Instruct-F16.gguf',836180256,'256f3a43bd4205ffef48d6b92715e1e70b5b0e9aef06522584967513a9985331'),
               ('mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf',453974304,'30ba2c7dd3127a4561b6cba9d13d0f711c91bdb38742e2f56d73c8cb596bd06d')]
    if args.size=='vl2b':
        repo='Qwen/Qwen3-VL-2B-Instruct-GGUF';rev='52d6c8ffea26cc873ac5ad116f8631268d7eb503'
        upstream='Qwen/Qwen3-VL-2B-Instruct';folder=ROOT/'models/qwen3-vl-2b-gguf'
        files=[('Qwen3VL-2B-Instruct-Q4_K_M.gguf',1107409952,'089d75c52f4b7ffc56ba998ffc50aae89fcafc755f9e7208aacca281dca6c2ae'),
               ('mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf',445053216,'f9a68fabba69c3b81e153367b2c7521030b0fa8bb0de400c9599c8e6725f9c82')]
    folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name,size,sha in files:
        path=folder/name
        if not path.exists():
            temp=path.with_suffix('.part')
            url=f'https://huggingface.co/{repo}/resolve/{rev}/{name}'
            if args.size.startswith('vl'):ranged_download(url,temp,size)
            else:
                with requests.get(url,stream=True,timeout=(30,120)) as response:
                    response.raise_for_status()
                    with temp.open('wb') as out:
                        for chunk in response.iter_content(8*1024*1024):out.write(chunk)
            with temp.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if temp.stat().st_size!=size or actual!=sha:raise RuntimeError('Candidate hash differs: '+name)
            temp.replace(path)
        with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=sha:raise RuntimeError('Existing file hash differs: '+name)
        rows.append(dict(file=name,bytes=size,sha256=sha));print(name+' verified',flush=True)
    (folder/'manifest.json').write_text(json.dumps(dict(repository=repo,revision=rev,
        upstream=upstream,license='Apache-2.0',files=rows),indent=2),encoding='utf-8')

if __name__=='__main__':main()

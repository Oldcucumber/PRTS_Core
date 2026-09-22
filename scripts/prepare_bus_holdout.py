"""Download a separately frozen Commons holdout with license metadata."""
import hashlib
import argparse
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--spec',type=Path,default=ROOT/'tests/fixtures/bus_holdout_v2.json')
    ap.add_argument('--manifest',default='holdout-v2-sources.json')
    a=ap.parse_args()
    cases=json.loads(a.spec.read_text(encoding='utf-8'))['cases']
    out=ROOT/'outputs/stage2/bus-generalization/data';out.mkdir(parents=True,exist_ok=True)
    titles='|'.join('File:'+c['file'] for c in cases)
    url='https://commons.wikimedia.org/w/api.php?'+urllib.parse.urlencode(dict(
        action='query',format='json',prop='imageinfo',iiprop='url|extmetadata|size',titles=titles))
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'PRTS-Core-research/0.2'}),timeout=30) as f:
        pages=json.load(f)['query']['pages'].values()
    infos={p['title'][5:]:p['imageinfo'][0] for p in pages}
    manifest=[]
    for c in cases:
        info=infos[c['file']];license_name=info['extmetadata']['LicenseShortName']['value']
        if not any(x in license_name for x in ('CC BY 2.0','CC BY 3.0','CC BY 4.0','CC BY-SA 2.0','CC BY-SA 2.5','CC BY-SA 3.0','CC BY-SA 4.0')):
            raise ValueError(f'Review license {license_name}')
        p=out/(c['id']+Path(c['file']).suffix)
        if not p.exists() or p.stat().st_size!=info['size']:
            req=urllib.request.Request(info['url'],headers={'User-Agent':'PRTS-Core-research/0.2'})
            with urllib.request.urlopen(req,timeout=60) as src,p.open('wb') as dst:
                while chunk:=src.read(1024*1024):dst.write(chunk)
        manifest.append(dict(c,source_metadata=info,local_file=str(p.relative_to(ROOT)),
                             sha256=hashlib.sha256(p.read_bytes()).hexdigest(),transforms='Original unchanged source'))
        (out/a.manifest).write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(dict(id=c['id'],license=license_name,bytes=p.stat().st_size)),flush=True)


if __name__=='__main__':main()

"""Download three explicitly licensed, real videos and preserve attribution metadata."""
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    cases = json.loads((ROOT/'tests/fixtures/bus_generalization.json').read_text(encoding='utf-8'))
    out = ROOT/'outputs/stage2/bus-generalization/data'
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for case in cases['cases']:
        query = urllib.parse.urlencode(dict(action='query', format='json', prop='imageinfo',
                                            iiprop='url|extmetadata|size', titles='File:'+case['file']))
        req = urllib.request.Request('https://commons.wikimedia.org/w/api.php?'+query,
                                     headers={'User-Agent':'PRTS-Core-research/0.2'})
        with urllib.request.urlopen(req, timeout=30) as response:
            info = next(iter(json.load(response)['query']['pages'].values()))['imageinfo'][0]
        license_name = info['extmetadata']['LicenseShortName']['value']
        if license_name not in ('CC BY 3.0', 'CC BY 4.0'):
            raise ValueError(f'Review changed license for {case["id"]}: {license_name}')
        path = out/(case['id']+'.webm')
        if not path.exists() or path.stat().st_size != info['size']:
            req = urllib.request.Request(info['url'], headers={'User-Agent':'PRTS-Core-research/0.2'})
            with urllib.request.urlopen(req, timeout=60) as src, path.open('wb') as dst:
                while chunk := src.read(1024*1024):
                    dst.write(chunk)
        record = dict(case, local_file=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      source_metadata=info, transforms='None; original video for independent annotation and evaluation.')
        manifest.append(record)
        print(json.dumps(dict(id=case['id'], bytes=path.stat().st_size, license=license_name), ensure_ascii=False), flush=True)
    (out/'sources.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

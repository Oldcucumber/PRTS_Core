"""Run the selected desktop core from a relocated integration bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case',choices=['brain','bus','notice','citybus20-brain','citybus20-other','private-bus-brain'],default='brain')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--provider',choices=['cpu','dml'],default='cpu')
    ap.add_argument('--tts',action='store_true',help='Windows offline Huihui voice; audio files, no speaker access')
    ap.add_argument('--verify-only',action='store_true')
    args=ap.parse_args()
    manifest=json.loads((ROOT/'BUNDLE.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        file=ROOT/entry['path']
        with file.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=entry['sha256']:raise ValueError('Bundle file differs: '+entry['path'])
    print(json.dumps(dict(verified_files=len(manifest['files']),status=manifest['status'])),flush=True)
    if args.verify_only:return
    profile=manifest['runtime']
    command=[sys.executable,'-X','utf8',str(ROOT/'scripts/replay_stream.py'),
             '--scenario',str(ROOT/'examples'/f'{args.case}.json'),'--output',str(args.output.resolve()),
             '--asr','sensevoice','--fps','2','--predecode',
             '--native-library',str(ROOT/profile['windows_vlm_library']),
             '--projector',Path(profile['projector']).name,
             '--onnx-semantic-dir',str(ROOT/profile['semantic_dir']),
             '--onnx-detector-dir',str(ROOT/profile['detector_dir']),
             '--onnx-provider','DmlExecutionProvider' if args.provider=='dml' else 'CPUExecutionProvider']
    if args.tts:command.append('--tts')
    if 'model_subdir' in profile:
        command+=['--model-subdir',profile['model_subdir'],'--language-file',profile['language_file'],
                  '--waiting-mode',profile['waiting_mode'],'--brain-strategy',profile['brain_strategy'],
                  '--sampling',profile['sampling'],'--native-threads',str(profile['native_threads']),
                  '--ocr-profile',profile.get('ocr_profile','bounded_upright')]
        if args.provider=='dml':command+=['--memory-library',str(ROOT/profile['memory_library'])]
    subprocess.run(command,cwd=ROOT,check=True)


if __name__=='__main__':main()

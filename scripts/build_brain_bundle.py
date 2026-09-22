"""Package the measured multimodal prototype with weights, evidence and Apple source."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--demo',type=Path,required=True)
    ap.add_argument('--validation',type=Path,required=True)
    ap.add_argument('--extra-run',type=Path,action='append',default=[])
    ap.add_argument('--extra-demo',type=Path,action='append',default=[])
    ap.add_argument('--include-known-demo-failures',action='store_true',
                    help='Package measured missed targets as explicit failed acceptance evidence; never mark them passed')
    ap.add_argument('--library',type=Path,default=Path('outputs/prts-native-official/libprts_vlm_lowmem.dll'))
    ap.add_argument('--language',default='Qwen3VL-4B-Instruct-Q4_K_M.gguf')
    ap.add_argument('--projector',default='mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf');ap.add_argument('--model-subdir',default='qwen3-vl-4b-gguf')
    a=ap.parse_args();out=a.output.resolve()
    if out.exists() or out.with_suffix('.zip').exists():raise FileExistsError(out)
    run=json.loads((a.run/'summary.json').read_text(encoding='utf-8'))
    implementation=json.loads((a.run/'implementation-sha256.json').read_text(encoding='utf-8'))
    for name,expected in implementation.items():
        if sha(ROOT/'prts_core'/name)!=expected:raise RuntimeError('Core changed after the selected replay: '+name)
    native_info=json.loads((a.run/'native-library.json').read_text(encoding='utf-8'))
    if sha(a.library)!=native_info['sha256']:raise RuntimeError('Selected runtime binary differs from replay')
    validation=json.loads(a.validation.read_text(encoding='utf-8'))
    if not validation['complete']:raise RuntimeError('Model validation is incomplete')
    events=[json.loads(s) for s in (a.run/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    if any(e['type'] in ('worker_error','request_error') for e in events):raise RuntimeError('Replay contains runtime errors')
    types={e['type'] for e in events}
    if not {'scene','brain_observation','answer','transcript','target_observed','wait_cancelled'}<=types:
        raise RuntimeError('Continuous replay lacks required model/interaction evidence')
    answers=[e for e in events if e['type']=='answer' and e.get('source')=='local_vision_language_model']
    if not answers or any(e.get('finish_reason')!='stop' for e in answers):raise RuntimeError('A complete real VLM answer is required')
    found=[e['wait']['target'] for e in events if e['type']=='target_observed']
    expected={'101','C002','B003'}
    if set(found)-expected or len(found)!=len(set(found)):raise RuntimeError('Unexpected or repeated target reminder')
    replay_acceptance=dict(passed=set(found)==expected,expected_targets=sorted(expected),
                           observed_targets=found,missed_targets=sorted(expected-set(found)))
    if not replay_acceptance['passed'] and not a.include_known_demo_failures:
        raise RuntimeError('Frozen replay goals were not met exactly once')
    if not run.get('memory',{}).get('dxgi_measured'):raise RuntimeError('Current complete-core DXGI memory evidence required')
    if run['memory']['samples']<(run['source_end_s']-run['source_start_s'])*5:raise RuntimeError('Memory sampling did not cover the replay')
    if not (a.demo/'demo.mp4').exists():raise FileNotFoundError(a.demo/'demo.mp4')
    out.mkdir(parents=True);sources={}
    def copy(src,dst=None):
        src=Path(src);src=src if src.is_absolute() else ROOT/src
        dst=Path(dst) if dst else src.relative_to(ROOT)
        target=out/dst;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
        sources[dst.as_posix()]=src.relative_to(ROOT).as_posix() if src.is_relative_to(ROOT) else 'User-provided local media'
    def tree(folder,suffixes=None):
        folder=Path(folder);folder=folder if folder.is_absolute() else ROOT/folder
        if not folder.is_dir():raise FileNotFoundError(folder)
        for file in sorted(folder.rglob('*')):
            if not file.is_file() or any(p in ('__pycache__','.build','Artifacts','input-frames') for p in file.parts):continue
            if file.name.endswith(('-raw.mp4','_raw.mp4')):continue
            if suffixes is None or file.suffix.lower() in suffixes or file.name in ("LICENSE","NOTICE"):copy(file)
    for folder in ('prts_core','native','apple','docs','licenses','tests','scripts'):
        tree(folder,{'.py','.swift','.h','.cpp','.c','.json','.md','.txt','.png','.jpg','.svg','.wav','.modulemap','.cmake','.sh','.js','.mjs'})
    for name in ('LICENSE','requirements-core-cpu.txt','requirements-core-dml.txt'):copy(name)
    copy('docs/BRAIN_BUNDLE_README.md','README.md')
    library_name=a.library.name
    copy(a.library,Path('bin/windows-x64')/library_name)
    copy('outputs/native-toolchain/prts_gpu_memory.dll','bin/windows-x64/prts_gpu_memory.dll')
    for model in (a.language,a.projector,'manifest.json'):copy(Path('models')/a.model_subdir/model)
    model_manifest=out/'models'/a.model_subdir/'manifest.json'
    selected=json.loads(model_manifest.read_text(encoding='utf-8'))
    selected['omitted_reference_variants']=[e['file'] for e in selected['files'] if e['file'] not in (a.language,a.projector)]
    selected['files']=[e for e in selected['files'] if e['file'] in (a.language,a.projector)]
    model_manifest.write_text(json.dumps(selected,indent=2),encoding='utf-8')
    for model in ('model.int8.onnx','tokens.txt','manifest.json'):copy(Path('models/sensevoice-int8')/model)
    for file in (ROOT/'.venv-dml/Lib/site-packages/rapidocr/models').glob('*.onnx'):copy(file,Path('models/ocr')/file.name)
    for src,dst,files in [
        ('outputs/stage2/perception/onnx/mask2former-1024-fp16-v2','models/perception/semantic',['semantic.onnx','manifest.json']),
        ('outputs/stage2/perception/onnx/yolo11n-rect','models/perception/detector',['detector.onnx','manifest.json'])]:
        for file in files:copy(Path(src)/file,Path(dst)/file)
    tree('outputs/stage2/brain',{'.json','.md','.py','.txt','.log','.png','.jpg'})
    tree('outputs/stage2/bus-generalization/data')
    for file in (ROOT/'outputs/stage2/bus-generalization/data').glob('*sources*.json'):copy(file,Path('media-sources')/file.name)
    tree('outputs/stage2/queue-holdout/data');tree('outputs/stage2/queue-holdout/data-v2')
    tree(a.run);tree(a.demo)
    for folder in a.extra_run+a.extra_demo:tree(folder)
    scenario=run['scenario'];source=Path(scenario['source']);source=source if source.is_absolute() else ROOT/source
    copy(source,'assets/brain/montage.mp4');copy(source.parent/'sources.json','assets/brain/sources.json')
    case=dict(scenario,source='assets/brain/montage.mp4')
    for entry in case['user_audio']:
        original=Path(entry['file']);copy(original,Path('assets/brain')/original.name);entry['file']='assets/brain/'+original.name
    examples=out/'examples';examples.mkdir(exist_ok=True)
    (examples/'brain.json').write_text(json.dumps(case,ensure_ascii=False,indent=2),encoding='utf-8')
    for name in ('bus','notice'):
        c=json.loads((ROOT/f'outputs/stage2/streaming/{name}-case.json').read_text(encoding='utf-8'))
        copy(c['source'],f'assets/{name}-source.mp4');c['source']=f'assets/{name}-source.mp4'
        for i,entry in enumerate(c.get('user_audio',[])):
            copy(entry['file'],f'assets/{name}-user-{i}.wav');entry['file']=f'assets/{name}-user-{i}.wav'
        (examples/f'{name}.json').write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding='utf-8')
    for extra in a.extra_run:
        extra_case=json.loads((extra/'summary.json').read_text(encoding='utf-8'))['scenario']
        src=Path(extra_case['source']);destination=Path('assets')/extra.name/src.name
        copy(src,destination);extra_case['source']=destination.as_posix()
        for entry in extra_case.get('user_audio',[]):
            src=Path(entry['file']);destination=Path('assets')/extra.name/src.name
            copy(src,destination);entry['file']=destination.as_posix()
        (examples/(extra.name.removesuffix('-v2')+'.json')).write_text(json.dumps(extra_case,ensure_ascii=False,indent=2),encoding='utf-8')
    for folder in ('outputs/stage2/maps/route-replay','outputs/stage2/perception/continuous-turn'):
        tree(folder,{'.json','.jsonl','.png','.jpg','.mp4','.md'})
    for file in ('outputs/stage2/perception/REPORT.md','outputs/stage2/perception/REPRODUCE.md'):copy(file)
    components=[]
    for file in sorted((out/'models').rglob('*')):
        if not file.is_file():continue
        components.append(dict(path=file.relative_to(out).as_posix(),bytes=file.stat().st_size,sha256=sha(file)))
    (out/'MODEL_MANIFEST.json').write_text(json.dumps(dict(
        candidate=a.model_subdir,language=a.language,projector=a.projector,components=components,
        notices='docs/STAGE2_THIRD_PARTY_NOTICES.md',apple_executed=False),indent=2),encoding='utf-8')
    secret_file=ROOT/'outputs/private/amap.key'
    secret=secret_file.read_bytes().strip() if secret_file.exists() else b''
    rows=[]
    for file in sorted(out.rglob('*')):
        if not file.is_file():continue
        if file.suffix in ('.key','.pem'):raise RuntimeError('Unexpected credential')
        if secret and file.stat().st_size<10_000_000 and secret in file.read_bytes():raise RuntimeError('Credential found in '+str(file.relative_to(out)))
        name=file.relative_to(out).as_posix()
        rows.append(dict(path=name,bytes=file.stat().st_size,sha256=sha(file),source=sources.get(name,'generated bundle metadata')))
    runtime=dict(windows_vlm_library='bin/windows-x64/'+library_name,
        memory_library='bin/windows-x64/prts_gpu_memory.dll',model_subdir=a.model_subdir,language_file=a.language,
        projector='models/'+a.model_subdir+'/'+a.projector,semantic_dir='models/perception/semantic',detector_dir='models/perception/detector',
        waiting_mode='brain',brain_strategy=run.get('brain_strategy','direct'),
        sampling=run['sampling'],
        native_threads=run['native_threads'],ocr_profile=json.loads((a.run/'run-arguments.json').read_text())['ocr_profile'],
        native_revision='b29c606e28a01b1bc8c1351026a0fa6e616bf6c4')
    manifest=dict(schema_version=2,status=('Runnable desktop multimodal prototype; Apple source uncompiled; see measured limits'
        if replay_acceptance['passed'] else 'Navigation revision with known failed interaction acceptance; see replay_acceptance and docs/REVISION_FREE_FORWARD.md'),
        runtime=runtime,validation=a.validation.relative_to(ROOT).as_posix() if a.validation.is_absolute() else a.validation.as_posix(),
        demonstrated_targets=found,replay_acceptance=replay_acceptance,memory=run['memory'],files=rows)
    (out/'BUNDLE.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',allowZip64=True) as z:
        for file in sorted(out.rglob('*')):
            if file.is_file():
                kind=zipfile.ZIP_STORED if file.suffix in ('.gguf','.onnx','.mp4','.webm') else zipfile.ZIP_DEFLATED
                z.write(file,Path(out.name)/file.relative_to(out),compress_type=kind,compresslevel=1)
    result=dict(file=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(rows)+1)
    archive.with_suffix('.zip.sha256.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()

"""Build an auditable research/Apple-source bundle from explicitly selected files.

The output is an integration candidate, not an Apple binary or field acceptance.
Run after the selected model's frozen regression and endurance have completed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def digest(file):
    with file.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve()
    if out.exists() or out.with_suffix('.zip').exists():raise FileExistsError(out)
    regression=json.loads((ROOT/'outputs/stage2/bus-generalization/vision-q8-regression-v1/results.json').read_text(encoding='utf-8'))
    if not regression['complete'] or len(regression['results'])!=7 or any(
            not r['positive_observed'] or r['false_positive_targets'] for r in regression['results']):
        raise RuntimeError('Selected Q8 bus development regression has not passed')
    holdout=json.loads((ROOT/'outputs/stage2/bus-generalization/vision-q8-holdout-v3/results.json').read_text(encoding='utf-8'))
    baseline=json.loads((ROOT/'outputs/stage2/bus-generalization/holdout-v3-first-pass/results.json').read_text(encoding='utf-8'))
    previous={r['case']['id']:r for r in baseline['results']}
    if not holdout['complete'] or len(holdout['results'])!=len(previous) or any(
        r['false_positive_targets'] or (previous[r['case']['id']]['positive_observed'] and not r['positive_observed'])
        for r in holdout['results']):
        raise RuntimeError('Selected Q8 retained cases regress; preserve first-pass failures')
    endurance_path=ROOT/'outputs/stage2/runtime/dml-queue-q8-20min/summary.json'
    endurance=json.loads(endurance_path.read_text(encoding='utf-8'))
    if endurance['duration_s']<1200 or endurance['with_assumed_frontend_2gb']>8_000_000_000:
        raise RuntimeError('Selected desktop memory gate has not passed')
    if any(endurance['events'].get(k,0) for k in ('worker_error','request_error')):
        raise RuntimeError('Selected endurance has worker errors')
    if not endurance.get('includes_queue'):raise RuntimeError('Selected endurance did not exercise queue OCR')
    decisions=json.loads((ROOT/'outputs/stage2/queue-holdout/bus-decisions-v2/results.json').read_text(encoding='utf-8'))
    if not decisions['passed']:raise RuntimeError('Current task rules changed cached bus decisions')
    queue_run=json.loads((ROOT/'outputs/stage2/streaming/queue-dml-v2/summary.json').read_text(encoding='utf-8'))
    targets=[e['wait']['target'] for e in queue_run['interesting_events'] if e['type']=='target_observed']
    if targets!=['215','494','B003'] or any(e['type'] in ('worker_error','request_error') for e in queue_run['interesting_events']):
        raise RuntimeError('Queue interaction replay did not match its frozen targets')
    out.mkdir(parents=True)
    provenance={}
    def copy(src,dst=None):
        src=Path(src);src=src if src.is_absolute() else ROOT/src
        dst=Path(dst) if dst else src.relative_to(ROOT)
        target=out/dst;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,target);provenance[dst.as_posix()]=str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else 'user-provided local media'
    def tree(folder,suffixes=None):
        for file in sorted((ROOT/folder).rglob('*')):
            if not file.is_file() or any(x in file.parts for x in ('__pycache__','.build','Artifacts')):continue
            if suffixes is None or file.suffix.lower() in suffixes or file.name=='LICENSE':copy(file)
    for folder in ('prts_core','native','apple','docs','licenses','tests'):
        tree(folder,{'.py','.swift','.h','.cpp','.c','.json','.md','.txt','.png','.jpg','.svg','.wav','.modulemap','.cmake','.js','.mjs'})
    for file in (ROOT/'scripts').iterdir():
        if file.is_file() and file.suffix in ('.py','.sh'):copy(file)
    for name in ('LICENSE','requirements-core-cpu.txt','requirements-core-dml.txt','requirements-offline.txt'):
        copy(name)
    copy('docs/INTEGRATION_BUNDLE_README.md','README.md')
    copy('outputs/prts-native-build/libprts_vlm.dll','bin/windows-x64/libprts_vlm.dll')
    for file in ('prts_geometry.dll','prts_onnx.dll','prts_image.dll','prts_queue.dll'):
        copy(Path('outputs/native-toolchain')/file,Path('bin/windows-x64')/file)
    model_files=[
        ('models/minicpm-v-4.6-gguf/MiniCPM-V-4_6-Q5_K_M.gguf',None),
        ('models/minicpm-v-4.6-gguf/mmproj-matrix-q8_0.gguf',None),
        ('models/minicpm-v-4.6-gguf/mmproj-matrix-q8_0-manifest.json',None),
        ('models/minicpm-v-4.6-gguf/manifest-Q5_K_M.json',None),
        ('models/sensevoice-int8/model.int8.onnx',None),('models/sensevoice-int8/tokens.txt',None),
        ('models/sensevoice-int8/manifest.json',None),
        ('outputs/stage2/perception/onnx/mask2former-1024-fp16-v2/semantic.onnx','models/perception/semantic/semantic.onnx'),
        ('outputs/stage2/perception/onnx/mask2former-1024-fp16-v2/manifest.json','models/perception/semantic/manifest.json'),
        ('outputs/stage2/perception/onnx/yolo11n-rect/detector.onnx','models/perception/detector/detector.onnx'),
        ('outputs/stage2/perception/onnx/yolo11n-rect/manifest.json','models/perception/detector/manifest.json')]
    for src,dst in model_files:copy(src,dst)
    for file in (ROOT/'.venv-dml/Lib/site-packages/rapidocr/models').glob('*.onnx'):
        copy(file,Path('models/ocr')/file.name)
    for name in ('bus','notice'):
        scenario=json.loads((ROOT/f'outputs/stage2/streaming/{name}-case.json').read_text(encoding='utf-8'))
        source=Path(scenario['source']);copy(source,f'assets/{name}-source.mp4');scenario['source']=f'assets/{name}-source.mp4'
        for i,entry in enumerate(scenario.get('user_audio',[])):
            audio=Path(entry['file']);copy(audio,f'assets/{name}-user-{i}.wav');entry['file']=f'assets/{name}-user-{i}.wav'
        folder=out/'examples';folder.mkdir(exist_ok=True)
        (folder/f'{name}.json').write_text(json.dumps(scenario,ensure_ascii=False,indent=2),encoding='utf-8')
    tree('outputs/stage2/bus-generalization/data')
    tree('outputs/stage2/queue-holdout')
    queue_input=ROOT/'outputs/stage2/streaming/queue-montage-input-v1'
    queue_case=json.loads((queue_input/'scenario.json').read_text(encoding='utf-8'))
    copy(queue_input/'montage.mp4','assets/queue/montage.mp4')
    copy(queue_input/'sources.json','assets/queue/sources.json')
    queue_case['source']='assets/queue/montage.mp4'
    for entry in queue_case['user_audio']:
        original=ROOT/entry['file'];destination='assets/queue/'+original.name
        copy(original,destination);entry['file']=destination
    for segment in queue_case['image_segments']:segment['file']=Path(segment['file']).relative_to(ROOT).as_posix()
    (out/'examples/queue.json').write_text(json.dumps(queue_case,ensure_ascii=False,indent=2),encoding='utf-8')
    city=dict(source='outputs/stage2/bus-generalization/data/commons-citybus-20.webm',start=0,end=30,
              text_commands=[dict(at=0,text='等20路公交车到了提醒我')],
              provenance='Commons Citybus 20 video; scripted text request. No synthetic camera or live microphone. Attribution in media-sources.')
    (out/'examples/citybus20.json').write_text(json.dumps(city,ensure_ascii=False,indent=2),encoding='utf-8')
    # Only relocate the private source path; preserve labels and media bytes.
    for file in (out/'tests/fixtures').glob('bus*.json'):
        doc=json.loads(file.read_text(encoding='utf-8'))
        for case in doc.get('cases',[]):
            if '132609' in case.get('source',''):case['source']='assets/bus-source.mp4'
        file.write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
    reports=[
        'bus-generalization/regression-v3','bus-generalization/holdout-v3-first-pass',
        'bus-generalization/holdout-v3-review','bus-generalization/vision-q8-review','bus-generalization/vision-q8-regression-v1',
        'bus-generalization/vision-q8-holdout-v3','bus-generalization/crop-probe-v1',
        'bus-generalization/latin-ocr-probe-v1','bus-generalization/medium-det-rec-probe-v1',
        'native-vlm/quantization-v2','native-vlm/vision-q8-v1',
        'native-image/model-v1','native-geometry/actual-masks-v1','native-onnx/semantic-v1',
        'apple/source-syntax-v3','perception/cpu-validation/dml-fp16-v2',
        'endurance/cpu-20min','endurance/dml-q5-20min','endurance/dml-vision-q8-20min',
        'runtime/dml-queue-q8-20min',
        'asr']
    for folder in reports:
        if not (ROOT/'outputs/stage2'/folder).exists():raise FileNotFoundError(folder)
        tree('outputs/stage2/'+folder,{'.json','.jsonl','.md','.py','.snapshot','.jpg','.png','.log'})
    for path in ('outputs/stage2/perception/REPORT.md','outputs/stage2/perception/REPRODUCE.md',
                 'outputs/stage2/perception/acceptance-freeze.json'):
        copy(path)
    for folder in ('notice-timeline','bus-dml-v5-timeline','bus-portable-miss','bus-dml-q8-v2-timeline','citybus20-dml-q8-v2-timeline','queue-dml-v1-timeline','queue-dml-v2-timeline'):
        tree('outputs/stage2/demos/'+folder,{'.mp4','.json','.jpg','.png'})
    for folder in ('bus-dml-q8-v1','citybus20-dml-q8-v1','queue-dml-v1','queue-dml-v2'):
        base=ROOT/'outputs/stage2/streaming'/folder
        for file in base.iterdir():
            if file.is_file() and file.name!='annotated_raw.mp4':copy(file)
        tree('outputs/stage2/streaming/'+folder+'/speech')
    tree('outputs/stage2/validation')
    tree('outputs/stage2/maps/route-replay',{'.mp4','.json','.jsonl','.png'})
    for file in ('source-time.labelled.mp4','processing-time.labelled.mp4','manifest.json','frame-000090.jpg'):
        copy(Path('outputs/stage2/perception/continuous-turn')/file)
    for file in (ROOT/'outputs/stage2/bus-generalization/data').glob('*sources*.json'):
        copy(file,Path('media-sources')/file.name)
    components=[]
    for file in sorted((out/'models').rglob('*')):
        if file.suffix not in ('.gguf','.onnx') and file.name!='tokens.txt':continue
        relative=file.relative_to(out).as_posix()
        if 'minicpm' in relative:component,license_id='MiniCPM-V-4.6','Apache-2.0'
        elif '/semantic/' in relative:component,license_id='Mask2Former Mapillary FP16 v2','CC-BY-NC-4.0'
        elif '/detector/' in relative:component,license_id='Ultralytics YOLO11n ONNX','AGPL-3.0'
        elif '/sensevoice-' in relative:component,license_id='SenseVoice INT8','FunASR Model Open Source License v1.1'
        else:component,license_id='RapidOCR PP-OCRv6 small','Apache-2.0'
        components.append(dict(component=component,path=relative,bytes=file.stat().st_size,
                               sha256=digest(file),license=license_id,apple_executed=False))
    (out/'MODEL_MANIFEST.json').write_text(json.dumps(dict(
        status='selected desktop research combination; Apple model validation pending',
        sources_and_notices='docs/STAGE2_THIRD_PARTY_NOTICES.md',components=components),indent=2),encoding='utf-8')
    # Actual credentials are checked locally and never printed or put in a manifest.
    private=ROOT/'outputs/private/amap.key'
    secret=private.read_bytes().strip() if private.exists() else b''
    rows=[]
    for file in sorted(out.rglob('*')):
        if not file.is_file():continue
        if file.suffix.lower() in ('.key','.pem'):raise RuntimeError('Unexpected credential file')
        if secret and file.stat().st_size<10_000_000 and secret in file.read_bytes():
            raise RuntimeError('Credential content found in '+str(file.relative_to(out)))
        relative=file.relative_to(out).as_posix()
        rows.append(dict(path=relative,bytes=file.stat().st_size,sha256=digest(file),
                         source=provenance.get(relative,'generated or relocated for this bundle')))
    manifest=dict(schema_version=1,status='research integration candidate; Apple source not compiled or executed',
                  runtime=dict(windows_vlm_library='bin/windows-x64/libprts_vlm.dll',
                               language='models/minicpm-v-4.6-gguf/MiniCPM-V-4_6-Q5_K_M.gguf',
                               projector='models/minicpm-v-4.6-gguf/mmproj-matrix-q8_0.gguf',
                               semantic_dir='models/perception/semantic',detector_dir='models/perception/detector'),
                  tested_desktop_envelope_bytes=endurance['with_assumed_frontend_2gb'],
                  endurance_report=str(endurance_path.relative_to(ROOT)),
                  endurance_implementation_sha256=endurance['implementation_sha256'],
                  endurance_scope='Frozen 20-minute mixed queue/bus/question/navigation workload; later field rejection and continuation-control refinements separately replayed.',
                  frontend_reservation_bytes=2_000_000_000,apple_executed=False,files=rows)
    (out/'BUNDLE.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for file in sorted(out.rglob('*')):
            if file.is_file():z.write(file,Path(out.name)/file.relative_to(out))
    result=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=digest(archive),files=len(rows),
                uncompressed_bytes=sum(r['bytes'] for r in rows),status=manifest['status'])
    archive.with_suffix('.sha256.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

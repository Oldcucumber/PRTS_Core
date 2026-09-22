"""Test global SenseVoice language/ITN settings without providing expected IDs."""
import gc
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import sherpa_onnx
import soundfile as sf
from prts_core.tasks import identifiers

out=ROOT/'outputs/stage2/asr/profiles';out.mkdir(parents=True,exist_ok=True)
source=ROOT/'outputs/cases';models=ROOT/'models/sensevoice-int8'
files=['bus_user.wav','crossing_user.wav','notice_user.wav','road_user.wav',
       'wait_number.wav','call_correct.wav','call_wrong.wav','cancel.wav','stop.wav']
expected={'bus_user.wav':'912','wait_number.wav':'A108','call_correct.wav':'A108','call_wrong.wav':'A180'}
frozen=[dict(file=f,sha256=hashlib.sha256((source/f).read_bytes()).hexdigest(),
             synthetic=not f.endswith('_user.wav'),expected_identifier=expected.get(f)) for f in files]
(out/'inputs.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
for language,itn in [('zh',True),('auto',True),('zh',False),('auto',False)]:
    recognizer=sherpa_onnx.OfflineRecognizer.from_sense_voice(model=str(models/'model.int8.onnx'),
        tokens=str(models/'tokens.txt'),num_threads=4,provider='cpu',language=language,use_itn=itn)
    rows=[]
    for case in frozen:
        audio,rate=sf.read(source/case['file'],dtype='float32',always_2d=True)
        stream=recognizer.create_stream();stream.accept_waveform(rate,audio.mean(1));recognizer.decode_stream(stream)
        text=stream.result.text;ids=identifiers(text)
        rows.append(dict(**case,text=text,identifiers=ids,
            expected_identifier_present=case['expected_identifier'] in ids if case['expected_identifier'] else None))
    tag=language+('-itn' if itn else '-raw')
    report=dict(language=language,itn=itn,rows=rows,all_identifier_cases_pass=all(r['expected_identifier_present'] is not False for r in rows))
    (out/f'{tag}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(tag,[(r['file'],r['text']) for r in rows if r['expected_identifier']])
    del recognizer,stream;gc.collect()

"""Task-quality comparison using existing local Whisper-small CPU weights.

Concurrent research may affect timings. No expected identifier enters decoding.
This CTranslate2 probe is not evidence of an Apple runtime.
"""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from faster_whisper import WhisperModel
from prts_core.tasks import identifiers
from prts_core.intent import route

out=ROOT/'outputs/stage2/asr/whisper-small';out.mkdir(parents=True,exist_ok=True)
cases=json.loads((ROOT/'outputs/stage2/asr/profiles/inputs.json').read_text(encoding='utf-8'))
model=WhisperModel(str(ROOT/'models/whisper-small'),device='cpu',compute_type='int8',cpu_threads=2,local_files_only=True)
rows=[]
for case in cases:
    ambient=case['file'].startswith('call_')
    prompt='这是一段环境广播。公交车到站、列车进站、叫号与窗口通知。' if ambient else '这是用户关于公交车、地铁、人行道导航、叫号提醒和读牌的语音命令。'
    segments,info=model.transcribe(str(ROOT/'outputs/cases'/case['file']),language='zh',beam_size=5,
        vad_filter=True,condition_on_previous_text=False,initial_prompt=prompt)
    text=''.join(s.text for s in segments);ids=identifiers(text)
    row=dict(**case,text=text,identifiers=ids,intent=None if ambient else route(text),
        expected_identifier_present=case['expected_identifier'] in ids if case['expected_identifier'] else None)
    rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
(out/'results.json').write_text(json.dumps(dict(model='local Whisper-small CTranslate2 INT8 CPU',
    status='Quality probe, not Apple deployment or isolated latency benchmark',rows=rows,
    all_identifier_cases_pass=all(r['expected_identifier_present'] is not False for r in rows)),ensure_ascii=False,indent=2),encoding='utf-8')

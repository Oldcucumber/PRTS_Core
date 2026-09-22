"""Target-blind queue screen interpretation candidate; not a production replacement."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
from prts_core.native_vlm import NativeVLM

PROMPT=('只读取画面中当前正在叫号或已可取餐的号码。不要把柜台编号、排队等待号码、'
        '广告、价格、汇率或车牌当成叫号。保留每个号码的字母、前导零和后缀；'
        '分开的号码分别列出，不猜测模糊文字。只输出 JSON：'
        '{"called":["可确认的完整号码"],"waiting":["尚未叫到的号码"],"uncertain":true或false}。'
        '如果没有可确认的叫号显示，called 是空数组。')


def main():
    out=ROOT/'outputs/stage2/queue-holdout/semantic-probe-v1'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    log=(out/'native-stderr.log').open('w');os.dup2(log.fileno(),2)
    data=(ROOT/'tests/fixtures/queue_holdout_v1.json').read_bytes()
    (out/'spec_snapshot.json').write_bytes(data);(out/'prompt.txt').write_text(PROMPT,encoding='utf-8')
    model=NativeVLM(ROOT/'models',ROOT/'outputs/prts-native-build/libprts_vlm.dll',image_slices=1,
                    language_file='MiniCPM-V-4_6-Q5_K_M.gguf',projector_file='mmproj-matrix-q8_0.gguf')
    rows=[]
    for case in json.loads(data)['cases']:
        file=ROOT/case['source']
        if hashlib.sha256(file.read_bytes()).hexdigest()!=case['sha256']:raise ValueError('Input changed')
        response=model.generate(PROMPT,cv2.imread(str(file)),max_tokens=150,raw_prompt=True)
        match=re.search(r'\{.*\}',response['text'],re.S)
        try:parsed=json.loads(match[0]) if match else None
        except json.JSONDecodeError:parsed=None
        row=dict(case=case['id'],response=response,parsed=parsed);rows.append(row)
        (out/'results.json').write_text(json.dumps(dict(status='diagnostic candidate; observed data is now development',results=rows),ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(row,ensure_ascii=False),flush=True)
    model.close()


if __name__=='__main__':main()

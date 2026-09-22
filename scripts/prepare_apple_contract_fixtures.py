"""Freeze Python reference outputs for the team's Swift Package tests.

Generating these files does not execute Swift or establish Apple model parity.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from prts_core.tasks import Tasks,identifiers
from prts_core.queue_evidence import called_in_audio,classify_rows
from prts_core.intent import route
from prts_core.navigation import Point,Destination,Step,WalkingRoute,RouteProgress,Location,bearing


def main():
    out=ROOT/'apple/PRTSCore/Tests/PRTSContractsTests/Resources'
    out.mkdir(exist_ok=True)
    inputs=['SL3 20A A108 B 二百零六 008 217 912','101X 101 0101','臨２８專１２夜九快一〇一',
            'R7A L3 57B 4833 PU 1611','一百零八 二百零六 十七 三号线','A 108 B 205','C-002 K–203 A 108 12-3']
    ids=[dict(input=t,expected=identifiers(t)) for t in inputs]
    spec=json.loads((ROOT/'tests/fixtures/intent_regression.json').read_text(encoding='utf-8'))
    intents=[dict(input=c['text'],expected=route(c['text'])) for c in spec['cases']]
    intents.extend(dict(input=t,expected=route(t)) for t in [
        '帮我等R7A路公交车','等夜9路到了提醒我','带我去公交站，到站后等20A路公交车',
        '我的号是Ｂ二百零六，叫到我时告诉我','不要停止导航，继续看路'])
    waiting=[]
    def evidence(value,vehicle='v1',role='route_display',**extra):
        return dict(text=value,score=.99,vehicle_label='bus',vehicle_id=vehicle,text_role=role,**extra)
    cases=[('R7A','',[dict(texts=[evidence('R7')],vehicles=[dict(label='bus',observation_id='v1')]),
                           dict(texts=[evidence('R7A')],vehicles=[dict(label='bus',observation_id='v1')]),
                           dict(texts=[evidence('R7A')],vehicles=[dict(label='bus',observation_id='v1')])]),
        ('20','机场',[dict(texts=[evidence('20'),evidence('火车站'),evidence('机场','v2')],vehicles=[dict(label='bus',observation_id='v1'),dict(label='bus',observation_id='v2')]),
                     dict(texts=[evidence('20'),evidence('往机场方向')],vehicles=[dict(label='bus',observation_id='v1')])]),
        ('4833','',[dict(texts=[evidence('4833',role='vehicle_body')],vehicles=[dict(label='bus',observation_id='v1')])]),
        ('28','',[dict(texts=[dict(evidence('28'),score=0.,route_candidate=True,route_verified=False,route_verification_attempted=True)],vehicles=[dict(label='bus',observation_id='v1')]),
                  dict(texts=[dict(evidence('28'),score=.55,route_verified=True,route_verification_attempted=True)],vehicles=[dict(label='bus',observation_id='v1')])])]
    for target,direction,observations in cases:
        state=Tasks();command=dict(action='wait',kind='bus',target=target,direction=direction);state.command(command,0)
        rows=[]
        for i,o in enumerate(observations,1):
            result=state.observe(o['texts'],o['vehicles'],i)
            rows.append(dict(**o,now=i,expected=result))
        waiting.append(dict(command=command,observations=rows))
    route_source=json.loads((ROOT/'outputs/stage2/maps/route-replay/route.json').read_text(encoding='utf-8'))
    d=route_source['destination']
    path=WalkingRoute(Destination(d['id'],d['name'],Point(**d['point']),d['address']),
                      [Step([Point(**p) for p in s['points']],s['instruction'],s['action'],s['road'],s['walk_type']) for s in route_source['steps']],route_source['provider'])
    tracker=RouteProgress(path);fixes=[]
    for i in range(math.ceil(tracker.length/2)+1):
        along=min(tracker.length,i*2)
        a,b,start,length,_=next((s for s in tracker.segments if s[2]+s[3]>=along),tracker.segments[-1])
        t=(along-start)/length;now=3+along/1.2
        fix=Location(Point(a.longitude+(b.longitude-a.longitude)*t,a.latitude+(b.latitude-a.latitude)*t),now,2,bearing(a,b),3)
        fixes.append(dict(input=asdict(fix),now=now,expected=tracker.update(fix,now)))
    queue_audio=[dict(input=t,expected=called_in_audio(t)) for t in [
        '请A108号到3号窗口办理','B 二百零六号请到12号柜台','请C-002取餐',
        '现在叫到008号，前往9号窗口','NOW SERVING R17 AT COUNTER 4',
        '请问A108有没有叫到','请问A108到哪里了','请A108等待','尚未叫到A108',
        '请到A108窗口办理','A108已经过号','这张票的号码是A108','窗口A108请到3号柜台','请A108']]
    queue_layouts=[]
    for name in ['layout-dev-v4','layout-dev-v3']:
        report=json.loads((ROOT/f'outputs/stage2/queue-holdout/{name}/results.json').read_text(encoding='utf-8'))
        for case in report['results']:
            entries=[{k:e[k] for k in ('text','score','box')} for e in case['ocr']]
            classified=classify_rows(entries)
            observations=[]
            for target in case['case']['positive_targets']+case['case']['negative_targets']:
                state=Tasks();state.command(dict(action='wait',kind='number',target=target),0)
                observations.append(dict(target=target,expected=state.observe(classified,[],1)))
            queue_layouts.append(dict(id=case['case']['id'],input=entries,roles=[e['queue_role'] for e in classified],observations=observations))
    source_files=['prts_core/tasks.py','prts_core/intent.py','prts_core/navigation.py','prts_core/queue_evidence.py']
    payload=dict(scope='Python-to-Swift contract parity only, not model accuracy; real recorded route plus synthetic fixes',
        identifiers=ids,intents=intents,waiting=waiting,route=asdict(path),route_fixes=fixes,
        queue_audio=queue_audio,queue_layouts=queue_layouts,
        reference_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_files})
    file=out/'python-parity.json';file.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(file=str(file),identifiers=len(ids),intents=len(intents),waiting=len(waiting),route_fixes=len(fixes),
                         queue_audio=len(queue_audio),queue_layouts=len(queue_layouts),swift_executed=False),ensure_ascii=False))


if __name__=='__main__':main()

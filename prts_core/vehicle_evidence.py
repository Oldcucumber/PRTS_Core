"""Vehicle-associated OCR with an explicit display/body hypothesis.

The upper-vehicle prior is a cheap candidate filter, not a trained destination
display detector. It can miss unusually placed signs and cannot certify service
direction. Retain boxes and scores so the hypothesis remains inspectable.
"""
import re
import cv2
import numpy as np
from .tasks import identifiers


def annotate_vehicle_text(entries,detection,image_size):
    w,h=image_size;x1,y1,x2,y2=np.asarray(detection['box'])*[w,h,w,h]
    result=[]
    for original in entries:
        e=dict(original);box=np.asarray(e['box']);cy=float(box[:,1].mean())
        relative_y=(cy-y1)/max(1,y2-y1)
        body=(relative_y>.62 or bool(re.search(r'SEATING|STANDEES|座位|企位|核载|核載|载客|車號|车号',e['text'],re.I)))
        e.update(vehicle_label=detection['label'],vehicle_id=detection['observation_id'],
                 crop_box=detection['box'],text_role='vehicle_body' if body else 'display_candidate',
                 display_hypothesis='relative_height_and_text_context',relative_vehicle_y=relative_y)
        result.append(e)
    return result


def read_vehicle(bgr,detection,read,verify_route=None):
    h,w=bgr.shape[:2]
    x1,y1,x2,y2=map(int,np.asarray(detection['box'])*[w,h,w,h])
    x1,y1=max(0,x1),max(0,y1);x2,y2=min(w,x2),min(h,y2)
    crop=bgr[y1:y2,x1:x2]
    if not crop.size:return []
    entries=[]
    for e in read(crop):
        entries.append(dict(e,box=(np.asarray(e['box'])+[x1,y1]).tolist()))
    entries=annotate_vehicle_text(entries,detection,(w,h))
    reliable=any(e['text_role']=='display_candidate' and e['score']>=.70 and identifiers(e['text']) for e in entries)
    if not reliable:
        # A close reading uses the sign-bearing upper crop at higher resolution.
        # No requested route is supplied to OCR, avoiding answer-conditioned text.
        top=crop[:max(1,round(crop.shape[0]*.62))]
        scale=min(3.,max(1.,960/top.shape[1]))
        zoom=cv2.resize(top,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
        refined=[]
        for e in read(zoom):
            refined.append(dict(e,box=(np.asarray(e['box'])/scale+[x1,y1]).tolist(),reading_pass='upper_crop'))
        entries.extend(annotate_vehicle_text(refined,detection,(w,h)))
    if verify_route and any(e['score']>=.45 for e in entries):
        verification=verify_route(crop)
        observed=verification.get('route_id','')
        for e in entries:
            e.update(route_verification_attempted=True,route_verification=verification,
                     route_verified=bool(observed and observed in identifiers(e['text']) and e['score']>=.45))
            if e['route_verified']:e['text_role']='route_display'
        if observed and not any(e['route_verified'] for e in entries):
            # Useful to ask for a closer view, but not a confirmed bus arrival.
            entries.append(dict(text=observed,score=0.,box=[[x1,y1],[x2,y1],[x2,y2],[x1,y2]],
                                vehicle_label=detection['label'],vehicle_id=detection['observation_id'],
                                route_candidate=True,route_verification_attempted=True,route_verified=False,
                                route_verification=verification,text_role='unconfirmed_route'))
    return entries


def verify_with_vlm(model,crop,vehicle_kind='bus'):
    # Target-independent reading: the requested route never enters this prompt.
    subject='这列车' if vehicle_kind=='train' else '这辆公交车'
    result=model.generate(subject+'线路显示屏上的线路号码是什么？只回答线路标识或无法确定。'
                          '车牌、车身编号、座位数都不是线路。',crop,max_tokens=40,raw_prompt=True)
    text=result['text'];found=identifiers(text)
    clear=(len(set(found))==1 and not re.search('无法|不确定|可能|看不清|没有|未能',text)
           and result.get('finish_reason','stop')=='stop')
    return dict(route_id=found[0] if clear else '',raw_answer=text,
                processing_ms=result.get('processing_ms'),method='target_independent_vlm_plus_ocr',
                model=result.get('model'))

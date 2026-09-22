"""One waiting task, navigation state and evidence-based deduplicated events."""
import re
import unicodedata


def identifiers(text):
    text=unicodedata.normalize('NFKC',text).upper().translate(str.maketrans('臨專','临专'))
    numerals={'零':0,'〇':0,'一':1,'幺':1,'二':2,'两':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
    def number(m):
        s=m.group(0)
        if not any(c in s for c in '十百千'):return ''.join(str(numerals[c]) for c in s)
        total=0;current=0
        for c in s:
            if c in numerals:current=numerals[c]
            else:total+=(current or 1)*{'十':10,'百':100,'千':1000}[c];current=0
        return str(total+current)
    text=re.sub('[零〇一幺二两三四五六七八九十百千]+',number,text)
    # ASR often separates the spoken letter from the number. Preserve independent
    # numeric fields, complete prefixes/suffixes and leading zeroes: SL3 != L3,
    # 20A != 20, and the separate texts "217 912" are not the number 217912.
    text=re.sub(r'(?<![A-Z])([A-Z]{1,2})\s+(?=\d)',r'\1',text)
    # Printed queue/route prefixes may use a separator: C-002 is C002, not 002.
    # Numeric compounds such as 12-3 remain separate identifiers for clarification.
    text=re.sub(r'(?<![A-Z])([A-Z]{1,4})\s*[-‐‑–]\s*(?=\d)',r'\1',text)
    return re.findall(r'(?<![A-Z0-9夜快特临专])[A-Z夜快特临专]*\d+[A-Z]*(?![A-Z0-9])',text)


def same_vehicle(entry,detection,all_detections):
    if entry.get('vehicle_label')!=detection['label']:return False
    expected=detection.get('observation_id')
    if expected is not None:return entry.get('vehicle_id')==expected
    # Compatibility for a single explicitly associated legacy crop. Never merge
    # unassociated text across two vehicles.
    return entry.get('vehicle_id') is None and sum(d['label']==detection['label'] for d in all_detections)==1


def direction_matches(requested,text):
    clean=lambda s:re.sub(r'[\s，,。./、→>\-]+','',unicodedata.normalize('NFKC',s).casefold())
    return not requested or clean(requested) in clean(text)


class Tasks:
    def __init__(self,navigating=False):
        self.navigating=navigating;self.wait=None;self.version=0

    def command(self,intent,now):
        if intent['action']=='continue_wait':
            if self.wait:
                text=(f'已经提醒过{self.wait["target"]}' if self.wait['notified'] else f'会继续留意{self.wait["target"]}')
                return 'wait_continued',text
            return 'need_target','请说明要等的线路或号码'
        action=intent['action'];self.version+=1
        if action=='navigate':self.navigating=True;return 'navigation_started','已开始观察人行道和障碍'
        if action=='stop':self.navigating=False;return 'navigation_stopped','已停止局部引导'
        if action=='cancel':self.wait=None;return 'wait_cancelled','已取消等待'
        if action=='wait':
            kind=str(intent.get('kind',''));target=str(intent.get('target','')).strip()
            if kind=='light':
                return 'unsupported_task','当前等待模块支持公交、列车和叫号；尚未接入持续灯色识别'
            if kind in ('bus','train','number') and identifiers(target):
                ids=list(dict.fromkeys(identifiers(target)))
                target=ids[0] if len(ids)==1 else ''
            if kind not in ('bus','train','number','light') or not target:
                return 'need_target','请说明要等的线路、号码或灯色'
            direction=str(intent.get('direction','')).strip()
            self.wait=dict(kind=kind,target=target,direction=direction,version=self.version,
                           started=now,notified=False,candidate_notified=False)
            suffix=f'往{direction}方向' if direction else ''
            return 'wait_started',f'正在等待{target}{suffix}，识别到后提醒'
        return 'question_received','正在查看当前画面'

    def observe(self,ocr,detections,now,audio_text='',audio_start=None):
        task=self.wait
        if not task or task['notified']:return None
        visual=' '.join(x['text'] for x in ocr if x['score']>=.70)
        target=unicodedata.normalize('NFKC',task['target']).upper()
        exact_visual=any(target in identifiers(x['text']) for x in ocr if x['score']>=.70)
        exact_audio=target in identifiers(audio_text)
        if audio_start is not None and audio_start<task['started']:exact_audio=False
        kind=task['kind'];source=None;selected_visual=visual;candidate=None;vehicle_id=None
        if kind=='number':
            from .queue_evidence import called_in_audio
            screen_words=bool(re.search('号|號|窗口|柜台|櫃台|请|請|就诊|就診|取餐',visual))
            negative_visual=bool(re.search('未叫|等待|候诊|候診|排队|排隊|取号|取號',visual))
            if exact_audio and target in called_in_audio(audio_text):source='audio'
            elif any(e.get('queue_verification_attempted') for e in ocr):
                for entry in ocr:
                    if entry['score']<.7 or target not in identifiers(entry['text']):continue
                    if entry.get('queue_role')=='called':
                        source='visual';selected_visual=entry['text'];break
                    if entry.get('queue_role')=='unconfirmed':candidate=entry['text']
            elif exact_visual and screen_words and not negative_visual:source='visual'
        elif kind in ('bus','train'):
            labels=('bus',) if kind=='bus' else ('train',)
            # Number must occur within the detected vehicle, not in an unrelated sign.
            for d in detections:
                if d['label'] not in labels:continue
                associated=[e for e in ocr if (e['score']>=.70 or e.get('route_verified') or e.get('route_candidate')) and same_vehicle(e,d,detections)]
                vehicle_text=' '.join(e['text'] for e in associated if e.get('text_role')!='vehicle_body')
                for entry in associated:
                    if target not in identifiers(entry['text']):continue
                    if entry.get('route_candidate'):
                        candidate=vehicle_text;continue
                    if entry.get('route_verification_attempted') and not entry.get('route_verified'):continue
                    if entry.get('text_role')=='vehicle_body':continue
                    if direction_matches(task.get('direction',''),vehicle_text):
                        source='visual';selected_visual=vehicle_text;vehicle_id=d.get('observation_id');break
                    candidate=vehicle_text
                if source:break
            if (exact_audio and re.search('到站|进站|進站|到达|到達',audio_text)
                    and not re.search('未|没|沒|即将|即將|将要|將要|还有|還有|不是|并非|並非',audio_text)
                    and direction_matches(task.get('direction',''),audio_text)):
                source='audio'
        # Light color needs a dedicated observation; OCR text alone cannot measure it.
        if source:
            task['notified']=True
            text=(f'广播中叫到了{target}' if kind=='number' and source=='audio' else
                  f'叫号屏显示{target}' if kind=='number' else
                  f'观察到目标{target}'+(f'，方向{task["direction"]}' if task.get('direction') else ''))
            return dict(type='target_observed',text=text,wait=dict(task),source=source,
                        observation_s=now,evidence_text=audio_text if source=='audio' else selected_visual,
                        vehicle_id=vehicle_id)
        if candidate and not task['candidate_notified']:
            task['candidate_notified']=True
            text=(f'看到了{target}，尚未确认是否已经叫到' if kind=='number' else
                  f'识别到{target}，尚未确认是否往{task["direction"]}方向' if task.get('direction')
                  else f'画面中可能是{target}，请调整相机以确认号码')
            return dict(type='target_candidate',text=text,
                        wait=dict(task),source='visual',observation_s=now,evidence_text=candidate)
        return None

    def apply_brain(self,decision,now):
        """The model owns semantic decisions; this method only owns task lifetime."""
        task=self.wait
        if not task or task['notified'] or decision['task_version']!=task['version']:return None
        if decision['decision']!='MATCH':return None
        task['notified']=True
        text=(f"观察到{task['target']}的叫号信息" if task['kind']=='number' else
              f"观察到{task['target']}路"+('公交车' if task['kind']=='bus' else '列车'))
        return dict(type='target_observed',text=text,wait=dict(task),
                    source='multimodal_waiting_brain',observation_s=now,
                    evidence_text=decision['observation'],brain_decision=decision)

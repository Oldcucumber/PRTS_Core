"""Called-number field tests; synthetic layouts do not measure OCR accuracy."""
import unittest
import cv2
import numpy as np
from prts_core.queue_evidence import called_in_audio,classify_rows,split_numeric_row
from prts_core.tasks import Tasks,identifiers
from prts_core.intent import route


def entry(text,x,y,w=90,h=20):
    return dict(text=text,score=.99,box=[[x,y],[x+w,y],[x+w,y+h],[x,y+h]])


class QueueEvidenceTests(unittest.TestCase):
    def wait(self,target):
        tasks=Tasks();tasks.command(dict(action='wait',kind='number',target=target),1)
        return tasks

    def test_audio_selects_ticket_not_service_counter(self):
        for speech,expected in [
            ('请A108号到3号窗口办理',['A108']),
            ('B 二百零六号请到12号柜台',['B206']),
            ('请C-002取餐',['C002']),
            ('现在叫到008号，前往9号窗口',['008']),
            ('NOW SERVING R17 AT COUNTER 4',['R17']),
        ]:
            with self.subTest(speech=speech):
                self.assertEqual(called_in_audio(speech),expected)
                for target in expected:
                    self.assertEqual(self.wait(target).observe([],[],2,speech,1.5)['type'],'target_observed')
                for target in set(identifiers(speech))-set(expected):
                    self.assertIsNone(self.wait(target).observe([],[],2,speech,1.5))

    def test_questions_negation_and_counter_instructions_are_not_calls(self):
        for speech in ['请问A108有没有叫到','请问A108到哪里了','请A108等待',
                       '尚未叫到A108','请到A108窗口办理','A108已经过号',
                       '这张票的号码是A108','窗口A108请到3号柜台','请A108']:
            with self.subTest(speech=speech):
                self.assertNotIn('A108',called_in_audio(speech))
                self.assertIsNone(self.wait('A108').observe([],[],2,speech,1.5))
        self.assertIsNone(self.wait('A108').observe([],[],2,'请A108到3号窗口',.5))

    def test_queue_roles_are_local_and_survive_camera_roll(self):
        rows=[entry('正在服务',10,10,100),entry('A108',10,50),
              entry('等待',160,10),entry('B206',160,50),
              entry('窗口',310,10),entry('3',310,50),
              entry('票号',460,10),entry('C002',460,50)]
        for angle in [0,18,-22]:
            rotation=cv2.getRotationMatrix2D((250,50),angle,1)
            transformed=[dict(e,box=cv2.transform(np.array([e['box']],np.float32),rotation)[0].tolist()) for e in rows]
            classified=classify_rows(transformed)
            self.assertEqual([e['queue_role'] for e in classified[1::2]],['called','waiting','counter','unconfirmed'])
            self.assertEqual(self.wait('A108').observe(classified,[],2)['type'],'target_observed')
            self.assertIsNone(self.wait('B206').observe(classified,[],2))
            self.assertIsNone(self.wait('3').observe(classified,[],2))
            self.assertEqual(self.wait('C002').observe(classified,[],2)['type'],'target_candidate')

    def test_prefix_separator_keeps_leading_zeroes(self):
        self.assertEqual(identifiers('C-002 K–203 A 108 12-3'),['C002','K203','A108','12','3'])
        self.assertEqual(self.wait('C-002').wait['target'],'C002')

    def test_ticket_under_numbered_counter_does_not_call_counter(self):
        rows=classify_rows([entry('NOMOR ANTRIAN',10,10,250),entry('B-031',20,50),
                           entry('LOKET 7',10,100),entry('D-024',10,130)])
        self.assertIsNone(self.wait('7').observe(rows,[],2))
        self.assertEqual(self.wait('D024').observe(rows,[],2)['type'],'target_observed')

    def test_mixed_ticket_counter_row_cannot_confirm_either_number(self):
        rows=classify_rows([entry('PICK-UP NUMBER',0,0,200),entry('739-8',0,40,200)])
        self.assertIsNone(self.wait('8').observe(rows,[],2))
        self.assertIsNone(self.wait('739').observe(rows,[],2))
        self.assertEqual(called_in_audio('请739-8到窗口办理'),[])

    def test_split_speech_tail_keeps_wait_identity_and_deduplication(self):
        task=self.wait('215');version=task.version
        action=route('到时提醒我',task.wait)
        self.assertEqual(action['action'],'continue_wait')
        self.assertEqual(task.command(action,2)[0],'wait_continued')
        self.assertEqual(task.version,version)
        self.assertEqual(task.wait['started'],1)
        self.assertEqual(task.observe([],[],3,'请215号取餐',2)['type'],'target_observed')
        task.command(route('叫到时再提醒我',task.wait),4)
        self.assertTrue(task.wait['notified'])
        self.assertIsNone(task.observe([],[],5,'请215号取餐',4))
        task.command(dict(action='cancel'),6)
        self.assertEqual(route('到时提醒我',task.wait)['action'],'wait')
        self.assertEqual(task.command(dict(action='continue_wait'),7)[0],'need_target')

    def test_visible_spacing_not_requested_target_or_fixed_digit_length(self):
        # Controlled glyph geometry, separate from photographs/recognition scores.
        image=np.zeros((40,240,3),np.uint8);starts=[3,15,27,62,74,120,132,144,156]
        for x in starts:cv2.rectangle(image,(x,5),(x+7,33),(255,255,255),-1)
        row=entry('734189502',0,0,180,39)
        self.assertEqual([e['text'] for e in split_numeric_row(image,row)],['734','18','9502'])
        self.assertEqual([e['text'] for e in split_numeric_row(image,dict(row,text='73418950'))],['73418950'])
        uniform=np.zeros_like(image)
        for x in range(3,3+12*9,12):cv2.rectangle(uniform,(x,5),(x+7,33),(255,255,255),-1)
        self.assertEqual([e['text'] for e in split_numeric_row(uniform,row)],['734189502'])


if __name__=='__main__':unittest.main()

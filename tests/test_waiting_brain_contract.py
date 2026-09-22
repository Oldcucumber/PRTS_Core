"""Tool-call identity contract; models still own interpretation and recognition."""
import json
import unittest
import numpy as np
from prts_core.waiting_brain import WaitingBrain

class ReplyModel:
    def __init__(self,ids,event='vehicle_present'):self.ids,self.event=ids,event
    def generate(self,*args,**kwargs):
        return dict(text=json.dumps(dict(observation='模型观察',observed_ids=self.ids,event_kind=self.event,
            decision='目标出现')),finish_reason='stop')

class BrainContractTests(unittest.TestCase):
    def call(self,target,observed,kind='bus',event='vehicle_present'):
        return WaitingBrain(ReplyModel(observed,event)).observe(np.zeros((20,20,3),np.uint8),
            dict(kind=kind,target=target,direction='',version=1),0)
    def test_suffixes_and_zeroes_are_distinct(self):
        self.assertEqual(self.call('101',['101X'])['decision'],'UNCLEAR')
        self.assertEqual(self.call('101X',['101X'])['decision'],'MATCH')
        self.assertEqual(self.call('43',['043'],'number','service_call')['decision'],'UNCLEAR')
    def test_mixed_numbers_cannot_be_split_into_an_event(self):
        self.assertEqual(self.call('8',['555-8'],'number','service_call')['decision'],'UNCLEAR')
        self.assertEqual(self.call('C002',['C-002'],'number','service_call')['decision'],'MATCH')
    def test_model_event_meaning_must_match_the_task(self):
        self.assertEqual(self.call('17',['17'],event='none')['decision'],'UNCLEAR')

    def test_environment_language_is_consumed_once_with_its_own_clock(self):
        class RecordingModel(ReplyModel):
            def __init__(self):super().__init__(['A108'],'service_call');self.images=[]
            def generate(self,prompt,image=None,**kwargs):
                self.images.append(image)
                return super().generate()
        model=RecordingModel();brain=WaitingBrain(model)
        task=dict(kind='number',target='A108',direction='',version=1)
        frame=np.zeros((20,20,3),np.uint8);ambient=[dict(text='请A108到3号窗口',observation_s=98.)]
        first=brain.observe(frame,task,100,ambient=ambient)
        second=brain.observe(frame,task,101,ambient=ambient)
        self.assertIsNone(model.images[0]);self.assertIsNotNone(model.images[1])
        self.assertEqual(first['evidence_s'],98.)
        self.assertEqual(first['observation_channel'],'environment_transcript')
        self.assertEqual(second['observation_channel'],'vision')

if __name__=='__main__':unittest.main()

"""Unseen identifiers and cross-vehicle negatives; logic tests, not OCR accuracy."""
import unittest
from prts_core.intent import route
from prts_core.tasks import Tasks,identifiers


def vehicle(identifier):return dict(label='bus',observation_id=identifier)
def text(value,vehicle_id='v1',role='route_display'):
    return dict(text=value,score=.99,vehicle_label='bus',vehicle_id=vehicle_id,text_role=role)


class GeneralizationTests(unittest.TestCase):
    def wait(self,target,direction=''):
        task=Tasks();task.command(dict(action='wait',kind='bus',target=target,direction=direction),0)
        return task

    def test_complete_identifiers_and_separated_numbers(self):
        self.assertEqual(identifiers('SL3 20A A108 B 二百零六 008 217 912'),
                         ['SL3','20A','A108','B206','008','217','912'])
        for target in ['20','466','8','20A','SL3','008','K203','R7A','夜9','专12','快101']:
            with self.subTest(target=target):
                self.assertEqual(route(f'等{target}路公交车到了告诉我')['target'],target)
                task=self.wait(target)
                self.assertEqual(task.wait['target'],target)
                self.assertIsNone(task.observe([text('9999')],[vehicle('v1')],1))
                self.assertEqual(task.observe([text(target)],[vehicle('v1')],2)['type'],'target_observed')

    def test_no_prefix_suffix_or_substring_match(self):
        for target,seen in [('20A','20'),('20','20A'),('8','18'),('466','46'),('SL3','L3'),('008','8'),('夜9','9'),('9','夜9'),('专12','12')]:
            with self.subTest(target=target,seen=seen):
                self.assertIsNone(self.wait(target).observe([text(seen)],[vehicle('v1')],1))

    def test_different_vehicle_cannot_supply_direction(self):
        task=self.wait('20','火车站');detections=[vehicle('v1'),vehicle('v2')]
        evidence=[text('20','v1'),text('机场','v1'),text('火车站','v2'),text('21','v2')]
        result=task.observe(evidence,detections,1)
        self.assertEqual(result['type'],'target_candidate')
        self.assertFalse(task.wait['notified'])
        self.assertIsNone(task.observe(evidence,detections,2))
        result=task.observe([text('20'),text('开往火车站')],detections,3)
        self.assertEqual(result['type'],'target_observed')
        self.assertEqual(result['vehicle_id'],'v1')

    def test_fleet_body_and_unrelated_sign_are_not_route(self):
        task=self.wait('4833')
        self.assertIsNone(task.observe([text('4833',role='vehicle_body')],[vehicle('v1')],1))
        self.assertIsNone(task.observe([dict(text='4833',score=.99)],[vehicle('v1')],2))
        self.assertIsNone(task.observe([text('4833','wrong_vehicle')],[vehicle('v1')],3))

    def test_vlm_candidate_cannot_confirm_without_ocr_agreement(self):
        task=self.wait('28')
        e=dict(text('28'),score=0.,route_candidate=True,route_verified=False,route_verification_attempted=True)
        self.assertEqual(task.observe([e],[vehicle('v1')],1)['type'],'target_candidate')
        self.assertFalse(task.wait['notified'])
        self.assertIsNone(task.observe([e],[vehicle('v1')],2))
        rejected=dict(text('28'),route_verification_attempted=True,route_verified=False)
        self.assertIsNone(task.observe([rejected],[vehicle('v1')],3))
        confirmed=dict(text('28'),score=.55,route_verification_attempted=True,route_verified=True)
        self.assertEqual(task.observe([confirmed],[vehicle('v1')],4)['type'],'target_observed')

    def test_replacement_cancel_and_directional_audio(self):
        task=self.wait('20')
        task.command(dict(action='wait',kind='bus',target='466',direction='机场'),2)
        self.assertIsNone(task.observe([text('20')],[vehicle('v1')],3))
        self.assertIsNone(task.observe([],[],3,'466路往火车站方向已到站',2.5))
        self.assertIsNone(task.observe([],[],3,'466路往机场方向即将到站',2.5))
        self.assertEqual(task.observe([],[],4,'466路往机场方向已到站',3)['type'],'target_observed')
        self.assertIsNone(task.observe([],[],5,'466路往机场方向已到站',4))
        task.command(dict(action='cancel'),6)
        self.assertIsNone(task.observe([],[],7,'466路往机场方向已到站',6))


if __name__=='__main__':unittest.main()

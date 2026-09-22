import unittest
import numpy as np
from prts_core.guidance import Guide
from prts_core.forward_obstacles import forward_scan
from prts_core.streaming import StreamingCore
from prts_core.interfaces import VideoFrame


class FreeForwardTests(unittest.TestCase):
    def observation(self):
        mask=np.zeros((100,160),bool)
        return dict(sidewalk_class=mask,sidewalk=mask.astype(np.float32),detections=[])

    def test_no_sidewalk_enters_free_mode_without_invented_path(self):
        r=Guide()(np.zeros((100,160,3),np.uint8),self.observation(),0)
        self.assertEqual((r['mode'],r['status'],r['path']),('free_forward','FREE',[]))
        self.assertEqual(r['forward_scan']['assessment'],'no_obstacle_detected')
        self.assertFalse(r['forward_scan']['metric_distance_available'])

    def test_forward_obstacle_alerts_but_side_and_overhead_do_not(self):
        for box,expected in [([.40,.45,.60,.95],True),([.01,.45,.15,.95],False),([.40,.05,.60,.25],False)]:
            p=self.observation();p['detections']=[dict(label='person',box=box,score=.9,class_id=0)]
            r=Guide()(np.zeros((100,160,3),np.uint8),p,0)
            self.assertEqual(r['status']=='STOP',expected)

    def test_semantic_only_fixed_obstacle_is_not_missed(self):
        p=self.observation();p['semantic_groups']=np.zeros((100,160),np.uint8)
        p['semantic_groups'][65:95,75:85]=3
        self.assertTrue(forward_scan(p)['obstacles'])

    def test_free_entry_once_and_new_obstacle_rearms_without_idle_nagging(self):
        now=[0.];events=[];core=StreamingCore(object(),events.append,clock=lambda:now[0])
        frame=VideoFrame(0,1,np.zeros((2,2,3),np.uint8))
        free=dict(mode='free_forward',status='FREE',direction='UNKNOWN',reason='free_forward',text='free',path=[])
        blocked=dict(free,status='STOP',reason='forward_obstacle',text='前方障碍')
        for i in range(20):now[0]=i*5.;core._publish_guidance(free,frame)
        self.assertEqual(sum(e['type']=='speech_request' for e in events),1)
        for i in range(10):now[0]+=5;core._publish_guidance(blocked,frame)
        self.assertEqual(sum(e['type']=='speech_request' for e in events),2)
        core._publish_guidance(free,frame);core._publish_guidance(blocked,frame)
        self.assertEqual(sum(e['type']=='speech_request' for e in events),3)
        core._publish_guidance(dict(mode='sidewalk',status='CANDIDATE',direction='FORWARD',reason='current_sidewalk',text='path'),frame)
        core._publish_guidance(free,frame)
        entries=[e for e in events if e['type']=='speech_request' and e['text'].startswith('进入自由前进')]
        self.assertEqual(len(entries),2)

if __name__=='__main__':unittest.main()

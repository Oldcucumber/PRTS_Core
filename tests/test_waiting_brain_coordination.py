"""Concurrency/ownership tests; controlled decisions are not model-quality evidence."""
import threading
import time
import unittest
import numpy as np
from prts_core.interfaces import VideoFrame
from prts_core.streaming import StreamingCore


class SlowBrain:
    independent_evidence=True
    brain_waiting=True
    def __init__(self):
        self.entered=threading.Event();self.release=threading.Event();self.ambient=[];self.hazard=False
    def observe(self,frame,**kw):
        return dict(guidance=dict(status='STOP' if self.hazard else 'CANDIDATE',direction='FORWARD',
                                 path=[],text='前方有障碍' if self.hazard else '沿路前行'),detections=[])
    def collect_evidence(self,*args):raise AssertionError('Expected independent brain worker')
    def collect_frame_evidence(self,frame,task):
        self.entered.set();self.release.wait(3)
        return dict(brain_decision=dict(task_version=task['version'],decision='MATCH',observation='模型确认目标出现'))
    def observe_ambient(self,text,stamp):self.ambient.append((text,stamp))
    def intent(self,text):return dict(action='ask')


class BrainCoordinationTests(unittest.TestCase):
    def wait_for(self,predicate):
        end=time.monotonic()+3
        while time.monotonic()<end:
            if predicate():return
            time.sleep(.01)
        self.fail('Timed out')
    def run_case(self,cancel):
        backend=SlowBrain();events=[];core=StreamingCore(backend,events.append).start()
        def frame(n):core.push_video(VideoFrame(time.monotonic(),n,np.zeros((20,20,3),np.uint8)))
        try:
            core.push_text('开始导航');core.push_text('等57B路公交车')
            self.wait_for(lambda:core.tasks.wait is not None);frame(1)
            self.assertTrue(backend.entered.wait(1))
            backend.hazard=True;frame(2)
            self.wait_for(lambda:any(e['type']=='speech_cancel' and e.get('reason')=='navigation_hazard' for e in events))
            self.assertFalse(backend.release.is_set())
            self.assertIsNotNone(core.tasks.wait)
            core.push_text('57B公交车到了',role='ambient');core.jobs.join()
            self.assertEqual(len(backend.ambient),1)
            self.assertFalse(any(e['type']=='target_observed' for e in events))
            if cancel:core.push_text('取消等待')
            backend.release.set()
            if cancel:
                core.close();self.assertFalse(any(e['type']=='target_observed' for e in events))
            else:
                self.wait_for(lambda:any(e['type']=='target_observed' for e in events));frame(3)
                core.close()
                found=[e for e in events if e['type']=='target_observed']
                self.assertEqual(len(found),1)
                self.assertEqual(found[0]['source'],'multimodal_waiting_brain')
                self.assertGreater(core.stats['frames_processed'],1)
        finally:
            backend.release.set()
            if core.running:core.close()
    def test_navigation_and_hazard_continue_while_brain_waits(self):self.run_case(False)
    def test_cancel_rejects_late_brain_decision(self):self.run_case(True)

if __name__=='__main__':unittest.main()

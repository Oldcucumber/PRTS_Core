"""Scheduling tests use controlled doubles; model quality requires recorded runs."""
import threading
import time
import unittest
import numpy as np
from prts_core.interfaces import VideoFrame, AudioChunk, AudioSegmenter
from prts_core.intent import route
from prts_core.streaming import StreamingCore


class Backend:
    def __init__(self):
        self.answer_started=threading.Event();self.release_answer=threading.Event()
    def observe(self,frame,**kw):
        return dict(guidance=dict(status='CANDIDATE',direction='FORWARD',path=[[.5,.9],[.5,.4]],text='沿人行道前行'),
                    detections=[],regions=[])
    def answer(self,text,frame,context):
        self.answer_started.set();self.release_answer.wait(3)
        return dict(text='测试答复',source='test_double')
    def transcribe(self,samples,role):return dict(text='停止导航',processing_ms=1)
    def intent(self,text):return dict(action='ask')


class StreamingTests(unittest.TestCase):
    def until(self,predicate,timeout=3):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if predicate():return
            time.sleep(.01)
        self.fail('Timed out awaiting event')

    def setUp(self):
        self.events=[];self.backend=Backend()
        self.core=StreamingCore(self.backend,self.events.append).start()

    def tearDown(self):
        self.backend.release_answer.set();self.core.close()

    def frame(self,seq):
        self.core.push_video(VideoFrame(time.monotonic(),seq,np.zeros((20,20,3),np.uint8)))

    def test_navigation_continues_during_question_and_stop_cancels_answer(self):
        self.core.push_text('开始导航');self.until(lambda:self.core.tasks.navigating)
        self.frame(1);self.until(lambda:self.core.stats['frames_processed']==1)
        self.core.push_text('前面有什么');self.assertTrue(self.backend.answer_started.wait(2))
        for i in range(2,7):self.frame(i);time.sleep(.015)
        self.assertGreater(self.core.stats['frames_processed'],1)
        self.core.push_text('停止导航')
        self.assertFalse(self.core.tasks.navigating)
        stop_sequence=max(e['sequence'] for e in self.events if e['type']=='navigation_stopped')
        self.backend.release_answer.set();self.frame(7)
        self.until(lambda:any(e['type']=='answer_cancelled' for e in self.events))
        self.assertFalse(any(e['type']=='guidance' and e['sequence']>stop_sequence for e in self.events))
        self.assertFalse(any(e['type']=='speech_request' and e['text']=='测试答复' for e in self.events))

    def test_audio_stop_is_recognized_while_answer_is_running(self):
        self.core.push_text('开始导航');self.until(lambda:self.core.tasks.navigating)
        self.frame(1);self.core.push_text('看看前面');self.assertTrue(self.backend.answer_started.wait(2))
        now=time.monotonic()
        self.core.push_audio(AudioChunk(now,0,np.ones(6400,np.float32)*.1,role='user',end_utterance=True))
        self.until(lambda:not self.core.tasks.navigating)
        self.assertFalse(self.backend.release_answer.is_set())

    def test_out_of_order_frames_and_latest_slot(self):
        self.frame(2);self.frame(1)
        self.assertEqual(self.core.frame.sequence,2)
        for i in range(3,80):self.frame(i)
        self.until(lambda:self.core.observation and self.core.observation[0].sequence==79)
        self.assertGreater(self.core.stats['frames_replaced'],0)

    def test_raw_playback_echo_suppressed(self):
        self.core.set_playback(True)
        self.core.push_audio(AudioChunk(time.monotonic(),0,np.ones(6400,np.float32)*.1,end_utterance=True))
        self.assertEqual(self.core.stats['echo_chunks_suppressed'],1)
        self.assertTrue(self.core.audio_jobs.empty())

    def test_queued_start_cannot_resurrect_navigation_after_stop(self):
        self.frame(1);self.core.push_text('看看前面');self.assertTrue(self.backend.answer_started.wait(2))
        self.core.push_text('开始导航');self.core.push_text('停止导航')
        self.backend.release_answer.set();self.core.jobs.join()
        self.assertFalse(self.core.tasks.navigating)

    def test_ambient_broadcast_does_not_create_user_task(self):
        self.core.push_text('请帮我等A108号叫号')
        self.until(lambda:self.core.tasks.wait is not None)
        self.core.push_text('请A180号到三号窗口',role='ambient')
        self.core.jobs.join();self.assertFalse(self.core.tasks.wait['notified'])
        self.core.push_text('请A108号到三号窗口',role='ambient')
        self.until(lambda:self.core.tasks.wait['notified'])
        self.assertEqual(sum(e['type']=='target_observed' for e in self.events),1)

    def test_slow_ground_does_not_hold_up_vehicle_wait(self):
        ground_started=threading.Event();release=threading.Event();events=[]
        class Independent(Backend):
            independent_evidence=True
            def observe(self,frame,**kw):
                ground_started.set();release.wait(2)
                return super().observe(frame,**kw)
            def collect_evidence(self,*a):raise AssertionError('Dense-scene path should not be used')
            def collect_frame_evidence(self,frame,wait):
                return dict(detections=[dict(label='bus',observation_id='17:0')],
                    ocr=[dict(text='57B',score=.99,vehicle_label='bus',vehicle_id='17:0',route_verified=True)])
        core=StreamingCore(Independent(),events.append).start()
        try:
            core.push_text('等57B路公交车')
            self.until(lambda:core.tasks.wait is not None)
            core.push_video(VideoFrame(time.monotonic(),17,np.zeros((20,20,3),np.uint8)))
            self.assertTrue(ground_started.wait(1))
            self.until(lambda:any(e['type']=='target_observed' for e in events))
            self.assertFalse(release.is_set())
            self.assertEqual(core.stats['frames_processed'],0)
        finally:release.set();core.close()

    def test_late_visual_answer_reports_age_in_speech(self):
        offset=[0.]
        self.core.clock=lambda:time.monotonic()+offset[0]
        self.frame(1);self.core.push_text('看看前面')
        self.assertTrue(self.backend.answer_started.wait(2))
        offset[0]=5.;self.backend.release_answer.set()
        self.until(lambda:any(e['type']=='answer' for e in self.events))
        answer=next(e for e in self.events if e['type']=='answer')
        self.assertTrue(answer['retrospective'])
        spoken=next(e for e in self.events if e['type']=='speech_request' and e['source']=='answer')
        self.assertIn('秒前的画面',spoken['text'])

    def test_coordinate_conversion_does_not_block_camera(self):
        from prts_core.navigation import Point,Location
        started=threading.Event();release=threading.Event()
        class Provider:
            def convert(self,point):
                started.set();release.wait(2)
                return Point(114,30,'GCJ02')
        self.core.navigation.provider=Provider()
        try:
            self.core.push_location(Location(Point(114,30,'WGS84'),time.monotonic(),3))
            self.assertTrue(started.wait(1));self.frame(1)
            self.until(lambda:self.core.stats['frames_processed']==1)
            self.assertIsNone(self.core.location)
            release.set();self.until(lambda:self.core.location is not None)
            self.assertEqual(self.core.location.point.crs,'GCJ02')
        finally:release.set()


class AudioTests(unittest.TestCase):
    def test_pcm_split_with_silence_and_discontinuity(self):
        vad=AudioSegmenter();chunks=[]
        for i in range(10):
            audio=np.ones(1600,np.float32)*(.08 if i<4 else 0)
            result=vad.feed(AudioChunk(i*.1,i,audio,role='user'))
            if result:chunks.append(result)
        self.assertEqual(len(chunks),1);self.assertAlmostEqual(chunks[0].timestamp_s,0)
        vad=AudioSegmenter()
        vad.feed(AudioChunk(0,0,np.ones(3200,np.float32)*.08))
        result=vad.feed(AudioChunk(1,9,np.ones(3200,np.float32)*.08,end_utterance=True))
        self.assertAlmostEqual(result.timestamp_s,1)

    def test_destination_and_confirmation_language(self):
        intent=route('带我去中心公交站，到了以后等17路公交到站提醒我')
        self.assertEqual(intent['query'],'中心公交站')
        self.assertEqual(intent['arrival_wait']['target'],'17')
        self.assertEqual(route('第二个',has_candidates=True)['index'],2)
        self.assertEqual(route('确认',has_candidates=True)['index'],0)
        self.assertEqual(route('等17路公交往火车站方向，来了提醒我')['direction'],'火车站')


if __name__=='__main__':unittest.main()

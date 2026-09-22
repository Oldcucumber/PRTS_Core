import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compose_stream_demo import playback_schedule


class PlaybackTests(unittest.TestCase):
    def test_audio_cannot_play_before_ready_and_cancel_truncates_it(self):
        e=dict(type='speech_request',sequence=1,emitted_s=10.,priority=30,expires_s=30.,replace_group='answer')
        ready=dict(sequence=1,file='a.wav',status='ready',audio_ready_s=12.,expires_s=30.,replace_group='answer')
        result=playback_schedule([e,dict(type='speech_cancel',sequence=2,emitted_s=14.)],[ready],{'a.wav':8.},10.)
        self.assertEqual((result[0]['start_s'],result[0]['end_s']),(2.,4.))
        # A result becoming ready after cancellation cannot resurrect the utterance.
        ready['audio_ready_s']=15.
        self.assertEqual(playback_schedule([e,dict(type='speech_cancel',sequence=2,emitted_s=14.)],[ready],{'a.wav':8.},10.),[])

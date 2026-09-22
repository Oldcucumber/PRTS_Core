"""Optional local speech consumer: synthesize when requested, not after replay.

WAV files and timing metadata are returned to a frontend/player. No physical
speaker is opened. This Windows reference is replaced by AVSpeechSynthesizer in
the Apple integration; it does not require a generative speech model.
"""
import json
from pathlib import Path
import queue
import threading
import time


class SpeechStream:
    def __init__(self, output, clock=time.monotonic):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.clock=clock;self.jobs=queue.PriorityQueue();self.generation=0
        self.group_latest={};self.entries=[];self.running=True;self.error=None
        self.thread=threading.Thread(target=self._run,name='prts-tts',daemon=True)
        self.thread.start()

    def accept(self,event):
        if event['type']=='speech_cancel':self.generation+=1
        if event['type']!='speech_request':return
        group=event['replace_group']
        self.group_latest[group]=event['sequence']
        self.jobs.put((-event['priority'],event['sequence'],self.generation,event))

    def _run(self):
        try:
            import comtypes
            import comtypes.client
            comtypes.CoInitialize()
            engine=comtypes.client.CreateObject('SAPI.SpVoice')
            voices=engine.GetVoices()
            voice=next((voices.Item(i) for i in range(voices.Count)
                        if 'HUIHUI' in voices.Item(i).Id.upper()),None)
            if voice is None:raise RuntimeError('Offline Chinese system voice unavailable')
            engine.Voice=voice;engine.Rate=0
            while self.running or not self.jobs.empty():
                try:_,_,generation,event=self.jobs.get(timeout=.1)
                except queue.Empty:continue
                try:
                    if generation!=self.generation or self.clock()>event['expires_s'] or self.group_latest[event['replace_group']]!=event['sequence']:
                        self.entries.append(dict(sequence=event['sequence'],status='cancelled_before_synthesis'));continue
                    path=self.output/f'speech_{event["sequence"]:06}.wav'
                    start=self.clock()
                    stream=comtypes.client.CreateObject('SAPI.SpFileStream')
                    stream.Open(str(path.resolve()),3,False)
                    engine.AudioOutputStream=stream
                    try:engine.Speak(event['text'],0)
                    finally:stream.Close()
                    if not path.is_file() or path.stat().st_size<=44:raise RuntimeError('Speech output was not generated')
                    status='ready' if generation==self.generation and self.clock()<=event['expires_s'] else 'cancelled_after_synthesis'
                    self.entries.append(dict(sequence=event['sequence'],text=event['text'],file=path.name,status=status,
                                             requested_s=event['emitted_s'],synthesis_started_s=start,
                                             audio_ready_s=self.clock(),expires_s=event['expires_s'],
                                             source=event['source'],replace_group=event['replace_group']))
                finally:self.jobs.task_done()
            engine=None;comtypes.CoUninitialize()
        except Exception as e:self.error=repr(e)

    def close(self):
        self.running=False;self.thread.join(30)
        if self.thread.is_alive():raise TimeoutError('Speech synthesis worker did not finish')
        (self.output/'manifest.json').write_text(json.dumps(dict(entries=self.entries,error=self.error,
            note='Generated during streaming. ready time is availability, not a measured physical speaker playback time.'),
            ensure_ascii=False,indent=2),encoding='utf-8')
        if self.error:raise RuntimeError(self.error)

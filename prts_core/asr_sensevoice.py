"""Portable CPU ASR candidate; identical ONNX model can use sherpa-onnx's C API."""
from pathlib import Path
import time
import numpy as np


class SenseVoiceASR:
    def __init__(self,model_dir,num_threads=4):
        import sherpa_onnx
        root=Path(model_dir)/'sensevoice-int8'
        self.recognizer=sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(root/'model.int8.onnx'),tokens=str(root/'tokens.txt'),
            num_threads=num_threads,provider='cpu',language='zh',use_itn=True)

    def __call__(self,audio,role='user'):
        rate=16000
        if isinstance(audio,(str,Path)):
            import soundfile as sf
            audio,rate=sf.read(audio,dtype='float32',always_2d=True)
            audio=audio.mean(1)
        start=time.perf_counter()
        stream=self.recognizer.create_stream()
        # sherpa-onnx resamples file inputs; the streaming PCM contract remains 16 kHz.
        stream.accept_waveform(rate,np.asarray(audio,dtype=np.float32))
        self.recognizer.decode_stream(stream)
        return dict(text=stream.result.text,model='sensevoice-int8-2025-09-09',
                    processing_ms=(time.perf_counter()-start)*1000,role=role)

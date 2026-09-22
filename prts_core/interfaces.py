"""In-process streaming contract, shared semantically with the native package."""
from dataclasses import dataclass
from typing import Any
import numpy as np

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SpatialSample:
    timestamp_s: float
    intrinsics: Any = None       # row-major 3x3, original upright image pixels
    camera_to_world: Any = None  # row-major 4x4, local right-handed, metres
    tracking: str = 'unavailable'
    depth_m: Any = None
    depth_confidence: Any = None
    mesh: Any = None
    world_frame_id: str = ''


@dataclass(frozen=True)
class VideoFrame:
    timestamp_s: float
    sequence: int
    bgr: np.ndarray               # upright image; caller does not mutate after push
    spatial: SpatialSample | None = None


@dataclass(frozen=True)
class AudioChunk:
    timestamp_s: float
    sequence: int
    samples: np.ndarray           # float32 mono, [-1, 1]
    sample_rate: int = 16000
    role: str = 'auto'            # user, ambient, auto
    end_utterance: bool = False
    echo_cancelled: bool = False


@dataclass(frozen=True)
class Utterance:
    timestamp_s: float
    end_s: float
    samples: np.ndarray
    role: str


class AudioSegmenter:
    """Small energy VAD with bounded storage and explicit utterance-end support.

    This is not speaker identification. Frontend PTT/voice processing can supply
    role and echo cancellation; continuous broadcast recognition uses ambient mode.
    """
    def __init__(self, threshold=.012, silence_s=.45, max_s=12., min_s=.18):
        self.threshold, self.silence_s = threshold, silence_s
        self.max_s, self.min_s = max_s, min_s
        self.last_sequence = None
        self.expected_time = None
        self.parts = []
        self.start = None
        self.end = 0.
        self.last_voice = 0.
        self.role = 'auto'
        self.preroll = []

    def reset(self):
        self.parts.clear();self.preroll.clear();self.start=None
        self.last_sequence=None;self.expected_time=None

    def feed(self, chunk):
        if chunk.sample_rate != 16000:
            raise ValueError('Resample incoming audio to 16000 Hz before push_audio')
        samples = np.asarray(chunk.samples, dtype=np.float32).reshape(-1)
        if len(samples) > 16000:
            raise ValueError('Send audio chunks of at most one second')
        if self.last_sequence is not None and (chunk.sequence != self.last_sequence+1 or
                abs(chunk.timestamp_s-self.expected_time) > .08):
            self.reset()
        self.last_sequence = chunk.sequence
        self.expected_time = chunk.timestamp_s+len(samples)/16000
        if self.start is not None and chunk.role != self.role:
            self.parts.clear();self.preroll.clear();self.start=None
        voiced = bool(len(samples) and np.sqrt(np.mean(samples*samples)) >= self.threshold)
        if self.start is None:
            self.preroll.append((chunk.timestamp_s, samples.copy()))
            while len(self.preroll)>1 and chunk.timestamp_s-self.preroll[0][0] > .18:
                self.preroll.pop(0)
            if not voiced:
                return None
            self.start = self.preroll[0][0]
            self.parts = [p[1] for p in self.preroll]
            self.preroll=[];self.role=chunk.role
        else:
            self.parts.append(samples.copy())
        self.end = self.expected_time
        if voiced:self.last_voice=self.end
        if chunk.end_utterance or self.end-self.last_voice >= self.silence_s or self.end-self.start >= self.max_s:
            result = Utterance(self.start, self.end, np.concatenate(self.parts), self.role)
            speech_duration = self.last_voice-self.start
            self.parts=[];self.start=None
            return result if speech_duration >= self.min_s else None
        return None

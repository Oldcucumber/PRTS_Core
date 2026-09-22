"""Deterministic short non-speech cues; no recordings or third-party samples."""
import array
import hashlib
import json
import math
from pathlib import Path
import wave
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'apple/PRTSCore/Sources/PRTSAppleModels/Resources/cues';out.mkdir(parents=True,exist_ok=True)
    rate=22050
    patterns={'target_found':[(660,.14),(0,.07),(880,.17)],
              'attention':[(440,.09),(0,.08),(440,.09),(0,.08),(440,.09)],
              'arrived':[(523.25,.12),(659.25,.12),(783.99,.18)]}
    rows=[]
    for name,notes in patterns.items():
        samples=[]
        for frequency,duration in notes:
            count=round(duration*rate)
            for i in range(count):
                envelope=min(1,i/(.015*rate),(count-1-i)/(.025*rate))
                samples.append(round(32767*.14*max(0,envelope)*math.sin(2*math.pi*frequency*i/rate)))
        file=out/(name+'.wav')
        with wave.open(str(file),'wb') as stream:
            stream.setparams((1,2,rate,len(samples),'NONE','not compressed'))
            stream.writeframes(array.array('h',samples).tobytes())
        rows.append(dict(cue=name,file=file.name,sample_rate=rate,channels=1,duration_s=len(samples)/rate,
                         sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    (out/'manifest.json').write_text(json.dumps(dict(origin='Generated sine tones; original project assets',
        meaning='Notification cues only; no crossing or boarding permission',files=rows),indent=2),encoding='utf-8')
    print(json.dumps(rows))


if __name__=='__main__':main()

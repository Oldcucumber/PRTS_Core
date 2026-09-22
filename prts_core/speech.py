"""Optional offline response WAV export; never plays audio automatically."""
import json
from pathlib import Path


def export_speech(events,output):
    import pyttsx3
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    engine=pyttsx3.init()
    voice=next((v.id for v in engine.getProperty('voices') if 'HUIHUI' in v.id.upper() or 'zh' in str(v.languages).lower()),None)
    if voice is None:raise RuntimeError('No offline Chinese system voice found. Install a Chinese voice or omit --tts.')
    engine.setProperty('voice',voice);engine.setProperty('rate',170)
    manifest=[]
    for i,event in enumerate(events):
        filename=f'response_{i:03}.wav'
        engine.save_to_file(event['text'],str(output/filename))
        manifest.append(dict(file=filename,text=event['text'],event_type=event['type'],
                             observation_s=event.get('observation_s'),voice=voice))
    engine.runAndWait()
    for entry in manifest:
        path=output/entry['file']
        if not path.is_file() or path.stat().st_size<=44:raise RuntimeError(f'TTS did not produce audio: {path}')
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    return manifest

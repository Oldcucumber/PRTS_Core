"""Local-only ASR audit. Inputs and derived audio stay on this computer."""
import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs', nargs='+', type=Path)
    p.add_argument('--model', type=Path, default=Path('models/whisper-small'))
    p.add_argument('--output', type=Path, default=Path('outputs/media-audit'))
    args = p.parse_args()
    if not (args.model / 'model.bin').is_file():
        p.error('ASR model must already be downloaded locally')
    from faster_whisper import WhisperModel
    model = WhisperModel(str(args.model), device='cpu', compute_type='int8', cpu_threads=6,
                         local_files_only=True)
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for i, source in enumerate(args.inputs, 1):
        metadata = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(source)
        ], encoding='utf-8'))
        wave = args.output / f'video_{i}.wav'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-vn',
                        '-ac', '1', '-ar', '16000', str(wave)], check=True)
        started = time.perf_counter()
        segments, info = model.transcribe(str(wave), language='zh', beam_size=5,
                                          vad_filter=True, word_timestamps=True,
                                          condition_on_previous_text=False)
        segments = [dict(start=s.start, end=s.end, text=s.text,
                         avg_logprob=s.avg_logprob, no_speech_prob=s.no_speech_prob,
                         words=[dict(start=w.start, end=w.end, word=w.word,
                                     probability=w.probability) for w in s.words or []]) for s in segments]
        report = dict(source=str(source.resolve()), metadata=metadata,
                      model=str(args.model.resolve()), local_only=True,
                      warning='Automatic transcript, not speaker diarization or human ground truth.',
                      seconds=time.perf_counter()-started, segments=segments)
        (args.output / f'video_{i}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        reports.append(report)
        print(json.dumps(dict(video=i, seconds=report['seconds'],
                              transcript=[{k:s[k] for k in ('start','end','text')} for s in segments]),
                         ensure_ascii=False), flush=True)
    (args.output / 'audit.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

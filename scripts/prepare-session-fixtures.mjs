// Windows-only test-fixture preparation. Synthesizes commands; never records users.
import {mkdir,readFile,writeFile} from 'node:fs/promises';import {execFileSync} from 'node:child_process';import {resolve} from 'node:path';
const dir=resolve('outputs/asr');await mkdir(dir,{recursive:true});
const phrases=['暂停导航','继续导航','静音','恢复播报','今天天气很好','向左调整'];
const ps="$voice=New-Object -ComObject SAPI.SpVoice; $voice.Voice=($voice.GetVoices() | Where-Object {$_.GetAttribute('Language') -match '804'} | Select-Object -First 1); "+phrases.map((text,i)=>"$stream=New-Object -ComObject SAPI.SpFileStream; $stream.Format.Type=18; $stream.Open('"+(dir+'/command-'+i+'.wav').replaceAll("'","''")+"',3,$false); $voice.AudioOutputStream=$stream; $voice.Speak('"+text+"') | Out-Null; $stream.Close();").join('');
execFileSync('powershell',['-NoProfile','-EncodedCommand',Buffer.from(ps,'utf16le').toString('base64')]);
function waveData(b){let p=12;while(p+8<=b.length){const n=b.readUInt32LE(p+4);if(b.toString('ascii',p,p+4)==='data')return b.subarray(p+8,p+8+n);p+=8+n+(n%2);}throw Error('No WAV samples');}
const parts=[Buffer.alloc(32000*10)];for(let i=0;i<4;i++){parts.push(waveData(await readFile(dir+'/command-'+i+'.wav')),Buffer.alloc(32000*8));}
const pcm=Buffer.concat(parts),h=Buffer.alloc(44);h.write('RIFF');h.writeUInt32LE(pcm.length+36,4);h.write('WAVEfmt ',8);h.writeUInt32LE(16,16);h.writeUInt16LE(1,20);h.writeUInt16LE(1,22);h.writeUInt32LE(16000,24);h.writeUInt32LE(32000,28);h.writeUInt16LE(2,32);h.writeUInt16LE(16,34);h.write('data',36);h.writeUInt32LE(pcm.length,40);await writeFile(dir+'/microphone.wav',Buffer.concat([h,pcm]));
const source=resolve(process.argv[2]||'VID20260919182406.mp4');
const ffmpeg=args=>execFileSync('ffmpeg',['-hide_banner','-loglevel','error','-y',...args]);
ffmpeg(['-i',source,'-t','4','-vf','scale=640:360,fps=10','-pix_fmt','yuv420p',dir+'/moving-camera.y4m']);
ffmpeg(['-i',source,'-frames:v','1','-vf','scale=640:360',dir+'/still.png']);
ffmpeg(['-loop','1','-i',dir+'/still.png','-t','2','-r','10','-pix_fmt','yuv420p',dir+'/camera.y4m']);
console.log('Prepared synthetic microphone and recorded-camera fixtures in outputs/asr; not public build assets.');

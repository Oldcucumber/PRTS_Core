import {NavigationSession,controlIntent} from './core-session.mjs';
import {SpeechOutput} from './speech-output.mjs';
import {AudioInput} from './audio-input.mjs';
const $=id=>document.getElementById(id),events=[];
const labels={initializing:'初始化',corridor:'通道跟随',free_forward:'自由前进',blocked:'前方受阻',interrupted:'感知中断',paused:'已暂停',ended:'已结束'};
function record(event){if(event.type==='scene'){window.prtsLatestScene=event;return;}const e={at_ms:performance.now(),...event};events.push(e);if(events.length>1000)events.shift();window.prtsEvents=events;
 if(e.type!=='scene')$('event-log').textContent=events.filter(x=>x.type!=='scene').slice(-12).map(x=>JSON.stringify(x)).join('\n');
 if(e.type==='playback_start'){$('speech-state').textContent='播放中';if(audio.ready)$('audio-state').textContent='播报期间不识别口令';audio.reset();}
 if(e.type==='playback_end'||e.type==='playback_cancel'){$('speech-state').textContent=speech.muted?'已静音':'语音就绪';setTimeout(()=>{if(audio.ready&&!speech.suppressed)$('audio-state').textContent='本地口令监听中';},350);}
 if(e.type==='playback_error')$('speech-state').textContent='语音不可用：'+e.reason;
 if(e.type==='audio_ready')$('audio-state').textContent='本地口令监听中';
 if(e.type==='audio_error')$('audio-state').textContent='麦克风／识别不可用：'+e.message+'；可在设置选择仅视觉调试';
 if(e.type==='audio_status')$('audio-state').textContent=e.message;
 if(e.type==='transcript'){const intent=controlIntent(e.text);record({type:'command',text:e.text,intent});if(intent==='pause')core.pause();else if(intent==='resume')core.resume();else if(intent==='mute')mute(true);else if(intent==='unmute')mute(false);}
}
const speech=new SpeechOutput(record),audio=new AudioInput(record,()=>speech.suppressed);
const core=new NavigationSession(event=>{record(event);if(event.type==='state'){$('nav-state').textContent=labels[event.state]||event.state;$('pause-nav').textContent=event.state==='paused'?'继续':'暂停';if(['paused','ended','interrupted'].includes(event.state))$('guidance-text').textContent='—';}if(event.type==='speech_request')$('guidance-text').textContent=event.text;speech.accept(event);});
function mute(value){speech.setMuted(value);$('mute-nav').textContent=value?'恢复播报':'静音';$('speech-state').textContent=value?'已静音':speech.localVoice()?'语音就绪':'本地中文音色不可用';record({type:'mute',value});}
function checkSound(){speech.accept({type:'speech_request',text:'导航语音已准备好',sequence:0,priority:20,observed_ms:performance.now(),expires_ms:performance.now()+10000,replace_group:'startup'});}
$('test-sound').onclick=checkSound;$('mute-nav').onclick=()=>mute(!speech.muted);$('pause-nav').onclick=()=>core.paused?core.resume():core.pause();
window.addEventListener('prts-start',()=>{core.start();$('pause-nav').disabled=false;checkSound();if(!$('visual-only').checked)void audio.start();else $('audio-state').textContent='仅视觉调试';});
window.addEventListener('prts-ready',()=>core.ready());
window.addEventListener('prts-stop',()=>{core.stop();void audio.stop();$('audio-state').textContent='麦克风已停止';$('pause-nav').disabled=true;});
window.addEventListener('prts-perception',e=>core.observe(e.detail));
setInterval(()=>{core.tick();$('debug-stats').textContent=JSON.stringify({...window.floorLabStats,samples:undefined,audioReady:audio.ready,localVoice:speech.localVoice()?.name,session:core.session,paused:core.paused},null,2);},250);
$('export-events').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify({schemaVersion:1,events},null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='prts-session.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
window.prtsSession={core,audio,speech};

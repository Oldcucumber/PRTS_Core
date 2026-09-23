// One utterance at a time; priority, expiry and cancellation precede synthesis.
export class SpeechOutput {
 constructor(emit,{synth=globalThis.speechSynthesis,Utterance=globalThis.SpeechSynthesisUtterance,clock=()=>performance.now()}={}){this.emit=emit;this.synth=synth;this.Utterance=Utterance;this.clock=clock;this.muted=false;this.current=null;this.pending=null;this.blockUntil=0;this.voice=null;}
 localVoice(){this.voice=this.synth?.getVoices().find(v=>v.localService&&/^zh(?:-|_)/i.test(v.lang));return this.voice;}
 cancel(){clearTimeout(this.timer);const old=this.current;this.current=null;this.pending=null;this.blockUntil=this.clock()+300;this.synth?.cancel();if(old)this.emit({type:'playback_cancel',sequence:old.event.sequence});}
 setMuted(value){this.muted=value;if(value)this.cancel();}
 accept(event){if(event.type==='speech_cancel'){this.cancel();return;}if(event.type!=='speech_request'||this.muted||event.expires_ms<this.clock())return;
 if(this.current){if(event.priority>this.current.event.priority)this.cancel();else{this.pending=event;return;}}
 this.play(event);
 }
 play(event){if(event.expires_ms<this.clock()||this.muted)return;
 if(!this.localVoice()||!this.Utterance){this.emit({type:'playback_error',reason:'未找到设备本地中文音色'});return;}
 const utterance=new this.Utterance(event.text);utterance.voice=this.voice;utterance.lang=this.voice.lang;utterance.rate=1.05;
 const item={event,utterance};this.current=item;this.blockUntil=Infinity;
 utterance.onstart=()=>{if(this.current===item)this.emit({type:'playback_start',sequence:event.sequence,text:event.text,observed_ms:event.observed_ms});};
 const finish=(type,reason)=>{if(this.current!==item)return;clearTimeout(this.timer);this.current=null;this.blockUntil=this.clock()+300;this.emit({type,sequence:event.sequence,reason});const next=this.pending;this.pending=null;if(next)this.play(next);};
 this.timer=setTimeout(()=>{if(this.current===item){this.cancel();this.emit({type:'playback_error',sequence:event.sequence,reason:'本地语音播放超时'});}},10000);this.timer.unref?.();
 utterance.onend=()=>finish('playback_end');utterance.onerror=e=>finish('playback_error',e.error);this.synth.speak(utterance);
 }
 get suppressed(){return this.clock()<this.blockUntil;}
}

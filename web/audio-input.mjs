export class AudioInput {
 constructor(emit,suppressed){this.emit=emit;this.suppressed=suppressed;this.generation=0;this.epoch=0;this.queue=[];this.busy=false;this.ready=false;this.gated=false;}
 async start(){const stopped=this.stop(),generation=this.generation;await stopped;if(generation!==this.generation)return;
 try{
 if(!crossOriginIsolated)throw new Error('浏览器隔离未启用，本地语音不可用；请刷新或使用支持隔离的浏览器');
 const stream=await navigator.mediaDevices.getUserMedia({video:false,audio:{channelCount:1,echoCancellation:true,noiseSuppression:true,autoGainControl:true}});
 if(generation!==this.generation){stream.getTracks().forEach(t=>t.stop());return;}this.stream=stream;
 stream.getTracks()[0].onended=()=>{if(generation===this.generation){this.emit({type:'audio_error',message:'麦克风已断开'});void this.stop();}};
 this.context=new AudioContext();await this.context.resume();await this.context.audioWorklet.addModule(new URL('./pcm-worklet.js',import.meta.url));if(generation!==this.generation)return;
 this.node=new AudioWorkletNode(this.context,'pcm-input');this.source=this.context.createMediaStreamSource(stream);this.gain=this.context.createGain();this.gain.gain.value=0;this.source.connect(this.node);this.node.connect(this.gain);this.gain.connect(this.context.destination);
 this.worker=new Worker(new URL('./asr.worker.js',import.meta.url));this.worker.onerror=e=>this.emit({type:'audio_error',message:e.message});
 this.worker.onmessage=({data})=>{if(generation!==this.generation)return;
 if(data.type==='ready'){this.ready=true;this.emit({type:'audio_ready'});}
 else if(data.type==='ack'){this.busy=false;this.pump();}
 else if(data.type==='transcript'){if(data.epoch===this.epoch&&!this.suppressed()&&performance.now()-data.observedAt<5000)this.emit(data);}
 else if(data.type==='error'){this.emit({type:'audio_error',message:data.message});void this.stop();}
 else this.emit({type:'audio_status',message:data.message});};
 this.node.port.onmessage=({data})=>{if(generation!==this.generation||!this.ready)return;
 if(this.suppressed()){if(!this.gated){this.reset();this.gated=true;}return;}this.gated=false;
 if(this.queue.length>=20){this.reset();this.emit({type:'audio_drop',message:'识别积压，已丢弃旧音频'});}
 this.queue.push({samples:data.samples,observedAt:performance.now(),epoch:this.epoch});this.pump();};this.worker.postMessage({type:'init'});this.emit({type:'audio_status',message:'加载本地语音模型'});
 }catch(e){if(generation===this.generation){this.emit({type:'audio_error',message:e.message});await this.stop();}}
 }
 reset(){this.epoch++;this.queue=[];this.worker?.postMessage({type:'reset',epoch:this.epoch});}
 pump(){if(this.busy||!this.ready||!this.queue.length)return;const packet=this.queue.shift();this.busy=true;this.worker.postMessage({type:'pcm',...packet},[packet.samples.buffer]);}
 async stop(){this.generation++;this.ready=false;this.busy=false;this.queue=[];this.worker?.terminate();this.worker=null;this.stream?.getTracks().forEach(t=>t.stop());this.stream=null;this.node?.disconnect();this.source?.disconnect();this.gain?.disconnect();const context=this.context;this.context=null;if(context&&context.state!=='closed')await context.close();}
}

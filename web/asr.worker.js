self.addEventListener('unhandledrejection',e=>postMessage({type:'error',message:String(e.reason)}));
// Classic worker: upstream Emscripten bundle expects a global Module.
var Module;let vad,recognizer,ready=false,pending=new Float32Array(0),epoch=0;
async function initialize(){
 const base=new URL('./speech/',self.location.href);const manifest=await (await fetch(new URL('manifest.json',base))).json();
 let bytes=new Uint8Array(manifest.bytes),offset=0;
 for(const part of manifest.chunks){const response=await fetch(new URL(part.name,base));if(!response.ok)throw Error('语音资源 '+response.status);const data=new Uint8Array(await response.arrayBuffer());if(data.length!==part.bytes)throw Error('语音资源不完整');bytes.set(data,offset);offset+=data.length;postMessage({type:'status',message:'语音模型 '+Math.round(offset/manifest.bytes*100)+'%'});}
 Module={mainScriptUrlOrBlob:new URL('sherpa-onnx-wasm-main-vad-asr.js',base).href,locateFile:path=>new URL(path,base).href,getPreloadedPackage:()=>bytes.buffer,print:()=>{},printErr:()=>{},onAbort:reason=>postMessage({type:'error',message:String(reason)}),onRuntimeInitialized(){try{
 vad=createVad(Module,{sileroVad:{model:'./silero_vad.onnx',threshold:.5,minSilenceDuration:.5,minSpeechDuration:.25,maxSpeechDuration:5,windowSize:512},sampleRate:16000,numThreads:1,provider:'cpu',debug:0});
 recognizer=new OfflineRecognizer({modelConfig:{tokens:'./tokens.txt',numThreads:1,debug:0,senseVoice:{model:'./sense-voice.onnx',language:'zh',useInverseTextNormalization:1}}},Module);
 bytes=null;ready=true;postMessage({type:'ready'});
 }catch(e){postMessage({type:'error',message:String(e)});}}};
 importScripts(new URL('sherpa-onnx-asr.js',base).href,new URL('sherpa-onnx-vad.js',base).href,new URL('sherpa-onnx-wasm-main-vad-asr.js',base).href);
}
onmessage=async({data})=>{try{
 if(data.type==='init'){await initialize();return;}if(!ready)return;
 if(data.type==='reset'){vad.reset();pending=new Float32Array(0);epoch=data.epoch;return;}
 if(data.type!=='pcm')return;
 if(data.epoch!==epoch){vad.reset();pending=new Float32Array(0);epoch=data.epoch;}
 const all=new Float32Array(pending.length+data.samples.length);all.set(pending);all.set(data.samples,pending.length);let i=0;
 for(;i+512<=all.length;i+=512)vad.acceptWaveform(all.subarray(i,i+512));pending=all.slice(i);
 while(!vad.isEmpty()){
 const segment=vad.front();vad.pop();const start=performance.now(),stream=recognizer.createStream();let result;
 try{stream.acceptWaveform(16000,segment.samples);recognizer.decode(stream);result=recognizer.getResult(stream);}finally{stream.free();}
 postMessage({type:'transcript',text:result.text,epoch,observedAt:data.observedAt,processingMs:performance.now()-start});
 }
 postMessage({type:'ack',epoch:data.epoch});
 }catch(e){postMessage({type:'error',message:String(e)});}};

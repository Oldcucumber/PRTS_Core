// Fractional resampling with a persistent phase, 100 ms PCM packets.
class PCMInput extends AudioWorkletProcessor {
 constructor(){super();this.phase=0;this.previous=0;this.samples=[];}
 process(inputs){const channels=inputs[0];if(!channels?.length)return true;const input=channels[0],step=sampleRate/16000;
 while(this.phase<input.length){const i=Math.floor(this.phase),fraction=this.phase-i;const a=i===0?this.previous:input[i-1],b=input[i];this.samples.push(a+(b-a)*fraction);this.phase+=step;if(this.samples.length===1600){const pcm=new Float32Array(this.samples);this.port.postMessage({samples:pcm,time:currentTime},[pcm.buffer]);this.samples=[];}}
 this.phase-=input.length;this.previous=input[input.length-1];return true;}
}
registerProcessor('pcm-input',PCMInput);

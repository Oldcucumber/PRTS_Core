import * as ort from './vendor/ort.webgpu.min.mjs';
import { analyzeFloor } from './corridor.mjs';

let session, backend, modelBuffer, config;
ort.env.wasm.wasmPaths = new URL('./vendor/',import.meta.url).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1,Math.min(4,navigator.hardwareConcurrency||2)) : 1;
ort.env.wasm.proxy = false;

async function create(backendName) {
  return ort.InferenceSession.create(modelBuffer, {executionProviders:[backendName], graphOptimizationLevel:'all'});
}

async function load({mode, profile}) {
  config = profile==='quality' ? {height:512,width:288} : {height:320,width:192};
  postMessage({type:'status',message:'正在加载本地模型…'});
  const response=await fetch(new URL(`./models/floor-${profile}.onnx`,import.meta.url));
  if(!response.ok) throw new Error(`模型加载失败（HTTP ${response.status}），请检查站点 models 目录。`);
  modelBuffer=new Uint8Array(await response.arrayBuffer());
  let reason='', adapterInfo='';
  if(mode!=='wasm') {
    try {
      if(!navigator.gpu) throw new Error('浏览器没有提供 WebGPU');
      const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
      if(!adapter) throw new Error('未找到可用 GPU adapter');
      const info=adapter.info;
      adapterInfo=info ? [info.vendor,info.architecture,info.description].filter(Boolean).join(' · ') : 'WebGPU adapter';
      if(info?.isFallbackAdapter || adapter.isFallbackAdapter) adapterInfo+=' (软件适配器)';
      backend='webgpu';
      postMessage({type:'status',message:'正在编译 GPU 推理管线…'});
      session=await create(backend);
      // Validate first execution too; unsupported GPU kernels can fail lazily.
      const tensor=new ort.Tensor('float32',new Float32Array(3*config.width*config.height),[1,3,config.height,config.width]);
      let outputs;
      try { outputs=await session.run({pixel_values:tensor}); }
      finally { tensor.dispose(); if(outputs) Object.values(outputs).forEach(t=>t.dispose()); }
    } catch(e) {
      if(mode==='webgpu') throw new Error(`WebGPU 启动失败：${e.message}。可切换「自动」或「WASM」。`);
      reason=String(e.message);
      if(session) await session.release().catch(()=>{});
      session=null;
    }
  }
  if(!session) {
    backend='wasm';
    postMessage({type:'status',message:reason ? 'GPU 不可用，正在切换 WASM…' : '正在启动 WASM 推理…'});
    session=await create(backend);
  }
  postMessage({type:'ready',backend,adapterInfo:backend==='webgpu'?adapterInfo:'CPU',reason,threads:ort.env.wasm.numThreads,...config});
}

async function infer(message) {
  const started=performance.now();
  const tensor=new ort.Tensor('float32',message.input,[1,3,config.height,config.width]);
  let outputs;
  try {
    try { outputs=await session.run({pixel_values:tensor}); }
    catch(e) {
      if(backend!=='webgpu' || message.mode!=='auto') throw e;
      await session.release().catch(()=>{});
      backend='wasm'; session=await create(backend);
      postMessage({type:'fallback',reason:String(e.message),backend,adapterInfo:'CPU'});
      outputs=await session.run({pixel_values:tensor});
    }
    const inferenceMs=performance.now()-started;
    const prob=outputs.floor_probability, classes=outputs.winning_class;
    const result=analyzeFloor(prob.data,classes.data,prob.dims[2],prob.dims[1],message.threshold);
    postMessage({type:'result',id:message.id,inferenceMs,totalMs:performance.now()-started,backend,...result},[result.region.buffer]);
  } finally { tensor.dispose(); if(outputs) Object.values(outputs).forEach(t=>t.dispose()); }
}

self.onmessage=async({data})=>{
  try {
    if(data.type==='init') await load(data);
    else if(data.type==='infer') await infer(data);
  } catch(e) { postMessage({type:'error',message:String(e.message||e)}); }
};

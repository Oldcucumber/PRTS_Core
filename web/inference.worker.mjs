import {decodeDetector,avoidDetections,localNavigation,semanticObstacles} from './navigation.mjs';
import * as ort from './vendor/ort.webgpu.min.mjs';
import { analyzeFloor } from './corridor.mjs';
import { unpadPrediction } from './camera-geometry.mjs?v=full-frame-1';
import { alignDepth, fuseDepth } from './depth-fusion.mjs?v=depth-1';

let session, backend, modelBuffer, config;
let detector=null,detectorBackend=null,detectorLabels=null;
let depthSession=null,depthBackend=null,depthBuffer=null;
ort.env.wasm.wasmPaths = new URL('./vendor/',import.meta.url).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1,Math.min(4,navigator.hardwareConcurrency||2)) : 1;
ort.env.wasm.proxy = false;

async function create(backendName) {
  return ort.InferenceSession.create(modelBuffer, {executionProviders:[backendName], graphOptimizationLevel:'all'});
}

async function load({mode, profile, depthEnabled=false}) {
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
  postMessage({type:'status',message:'正在加载 YOLO11n 障碍检测…'});
  const detectorBytes=await (await fetch(new URL('./models/detector.onnx',import.meta.url))).arrayBuffer();
  detectorLabels=(await (await fetch(new URL('./models/detector-manifest.json',import.meta.url))).json()).labels;
  async function initDetector(provider){
    detectorBackend=provider;detector=await ort.InferenceSession.create(detectorBytes,{executionProviders:[provider]});
    const t=new ort.Tensor('float32',new Float32Array(3*384*640),[1,3,384,640]);let out;
    try{out=await detector.run({rgb:t});}finally{t.dispose();if(out)Object.values(out).forEach(x=>x.dispose());}
  }
  try{await initDetector(backend);}catch(e){if(mode!=='auto'||backend!=='webgpu')throw e;await detector?.release().catch(()=>{});await initDetector('wasm');}
  if(depthEnabled){
    postMessage({type:'status',message:'正在加载深度模型 Depth Anything V2 Small…'});
    const response=await fetch(new URL('./models/depth-small.onnx',import.meta.url));
    if(!response.ok)throw new Error(`深度模型加载失败（HTTP ${response.status}）`);
    depthBuffer=new Uint8Array(await response.arrayBuffer());
    try{await createDepth(backend);}
    catch(error){
      if(mode!=='auto'||backend!=='webgpu')throw new Error(`深度模型启动失败：${error.message}`);
      await depthSession?.release().catch(()=>{});depthSession=null;
      reason+=" 深度模型回退 WASM。";await createDepth('wasm');
    }
  }
  postMessage({type:'ready',backend,detectorBackend,depthBackend,depthSize:252,depthEnabled,adapterInfo:backend==='webgpu'?adapterInfo:'CPU',reason,threads:ort.env.wasm.numThreads,...config});
}

async function createDepth(provider){
  depthBackend=provider;
  depthSession=await ort.InferenceSession.create(depthBuffer,{executionProviders:[provider],graphOptimizationLevel:'all'});
  const tensor=new ort.Tensor('float32',new Float32Array(3*252*252),[1,3,252,252]);
  let outputs;
  try{outputs=await depthSession.run({pixel_values:tensor});}
  finally{tensor.dispose();if(outputs)Object.values(outputs).forEach(t=>t.dispose());}
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
    const segmentationMs=performance.now()-started;
    const prob=outputs.floor_probability, classes=outputs.winning_class;
    const frame=unpadPrediction(outputs.semantic_confidence.data,classes.data,prob.dims[2],prob.dims[1],message.contentRect);
    const semanticClasses=frame.classes.slice();
    // ADE20K: floor=3, sidewalk=11, path=52. Road=6 is never a walking corridor.
    frame.classes=frame.classes.map(x=>[3,11,52].includes(x)?3:0);
    const baseline=analyzeFloor(frame.probability,frame.classes,frame.width,frame.height,message.threshold);
    let result=baseline,depth=null,depthMs=0;
    if(depthSession){
      const t=performance.now();
      const depthTensor=new ort.Tensor('float32',message.depthInput,[1,3,252,252]);
      let output;
      try{
        try{output=await depthSession.run({pixel_values:depthTensor});}
        catch(error){
          if(depthBackend!=='webgpu'||message.mode!=='auto')throw error;
          await depthSession.release().catch(()=>{});depthSession=null;await createDepth('wasm');
          postMessage({type:'status',message:'深度 GPU 运行失败，深度已切换 WASM'});
          output=await depthSession.run({pixel_values:depthTensor});
        }
        depthMs=performance.now()-t;
        const prediction=output.relative_depth;
        const aligned=alignDepth(prediction.data,prediction.dims[2],prediction.dims[1],message.depthContentRect,frame.width,frame.height);
        const fused=fuseDepth(frame,baseline,aligned,message.threshold);
        result=fused.result;
        depth={heat:fused.heat,excluded:fused.excluded,removedFraction:fused.removedFraction,fit:fused.fit,backend:depthBackend};
      }finally{depthTensor.dispose();if(output)Object.values(output).forEach(t=>t.dispose());}
    }
    const detectorStart=performance.now();let detOut;
    const detTensor=new ort.Tensor('float32',message.detectorInput,[1,3,384,640]);let detections;
    try{detOut=await detector.run({rgb:detTensor});const prediction=Object.values(detOut)[0];detections=decodeDetector(prediction.data,prediction.dims,message.detectorRect,detectorLabels);}
    finally{detTensor.dispose();if(detOut)Object.values(detOut).forEach(x=>x.dispose());}
    const detectorMs=performance.now()-detectorStart;
    const constrained={...frame,classes:frame.classes.map((x,i)=>result.region[i]?x:0)};
    result=avoidDetections(constrained,result,detections,message.threshold);
    const fixed=semanticObstacles(semanticClasses,frame.probability,frame.width,frame.height,message.halfAngle||15);
    const nav=localNavigation(result,baseline,[...detections,...fixed],message.halfAngle||15);
    const transfers=new Set([result.region.buffer]);
    if(depth){transfers.add(baseline.region.buffer);transfers.add(depth.heat.buffer);transfers.add(depth.excluded.buffer);}
    postMessage({type:'result',id:message.id,inferenceMs:segmentationMs+depthMs+detectorMs,segmentationMs,depthMs,detectorMs,detectorBackend,detections,nav,semanticClasses,totalMs:performance.now()-started,backend,...result,baseline:depth?baseline:null,depth},[...transfers]);
  } finally { tensor.dispose(); if(outputs) Object.values(outputs).forEach(t=>t.dispose()); }
}

self.onmessage=async({data})=>{
  try {
    if(data.type==='init') await load(data);
    else if(data.type==='infer') await infer(data);
  } catch(e) { postMessage({type:'error',message:String(e.message||e)}); }
};

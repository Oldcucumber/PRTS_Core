import {coverCrop,cameraConstraints} from './camera-geometry.mjs';

const $=id=>document.getElementById(id);
const view=document.querySelector('.camera-view');
const video=$('source'), canvas=$('result'), ctx=canvas.getContext('2d');
const capture=document.createElement('canvas'), captureCtx=capture.getContext('2d');
const resize=document.createElement('canvas'), resizeCtx=resize.getContext('2d',{willReadFrequently:true});
const maskCanvas=document.createElement('canvas'), maskCtx=maskCanvas.getContext('2d');
const dialog=$('settings-dialog');
let options={source:'camera',backend:'auto',profile:'fast',threshold:.55,file:null};
let draftFile=null;
let worker=null,stream=null,objectURL=null,generation=0,frameId=0,running=false,busy=false;
let config=null,capturedAt=0,lastFrameTime=-1,timer=null,lastResultAt=0,fps=0;
let inputPromise=null,inputReady=false;
let viewRevision=0,capturedViewRevision=0;
const stats={frames:0,backend:null,direction:'UNKNOWN',inferenceMs:0,totalMs:0,fps:0,errors:[],ready:false,samples:[]};
window.floorLabStats=stats;

function status(message){$('status').textContent=message;}
function updateConfig(){
  view.classList.toggle('camera-input',options.source==='camera');
  $('source-label').textContent=options.source==='camera'?'后置摄像头':options.source==='sample'?'内置测试视频':options.file?.name||'本地视频';
  $('config-summary').textContent=`${options.profile==='fast'?'192 × 320':'288 × 512'} · 阈值 ${options.threshold.toFixed(2)}`;
  if(!stats.ready){
    $('backend-badge').textContent=options.backend==='auto'?'AUTO':options.backend==='webgpu'?'WebGPU':'WASM';
    $('device').textContent=`SegFormer-B0 · ${options.backend==='auto'?'WebGPU 优先':options.backend==='webgpu'?'GPU':'CPU'}`;
  }
}
function runButton(active){
  $('start-label').textContent=active?'停止':'开始';
  $('start').classList.toggle('active',active);
  $('start-icon').innerHTML=active?'<rect x="5" y="5" width="10" height="10" rx="1"/>':'<path d="M5 3.5v13L16 10z"/>';
}
function stop(message='已停止'){
  generation++;running=false;busy=false;clearTimeout(timer);
  worker?.terminate();worker=null;
  stream?.getTracks().forEach(t=>t.stop());stream=null;
  video.pause();video.srcObject=null;video.removeAttribute('src');video.load();
  if(objectURL)URL.revokeObjectURL(objectURL);objectURL=null;
  config=null;inputPromise=null;inputReady=false;stats.ready=false;
  stats.direction='UNKNOWN';stats.fps=0;
  canvas.style.display='none';video.style.visibility='visible';$('empty').hidden=false;
  $('preview-state').textContent=options.source==='camera'?'摄像头已停止':'视频已停止';
  $('run-state').textContent='待机';$('direction').textContent='—';
  $('fps').textContent='—';$('latency').textContent='—';$('total').textContent='—';
  runButton(false);document.body.classList.remove('running');updateConfig();status(message);
}
function fail(message){stats.errors.push(message);stop(message);$('run-state').textContent='错误';$('preview-state').textContent='输入或推理不可用';}
function resetStats(){
  Object.assign(stats,{frames:0,backend:null,direction:'UNKNOWN',inferenceMs:0,totalMs:0,fps:0,errors:[],ready:false,samples:[]});
  lastResultAt=0;fps=0;lastFrameTime=-1;$('frames').textContent='0';$('fps').textContent='—';$('latency').textContent='—';$('total').textContent='—';
}
function inputError(error){
  if(error.name==='NotAllowedError')return '摄像头权限被拒绝。可在浏览器中授权后重试，或在设置中切换视频输入。';
  if(error.name==='NotFoundError')return '未检测到摄像头。可在设置中切换视频输入。';
  if(error.name==='NotReadableError')return '摄像头不可读取，可能被其他应用占用。';
  return String(error.message||error);
}

function ensureInput(){
  if(inputReady)return Promise.resolve(true);
  if(inputPromise)return inputPromise;
  const token=generation;
  $('run-state').textContent='连接中';
  $('preview-state').textContent=options.source==='camera'?'正在连接摄像头':'正在加载视频';
  status(options.source==='camera'?'摄像头初始化':'视频初始化');
  inputPromise=Promise.resolve().then(async()=>{
    try{
      if(options.source==='camera'){
        if(!isSecureContext||!navigator.mediaDevices?.getUserMedia)throw new Error('摄像头需要 HTTPS 或本机 localhost。');
        const newStream=await navigator.mediaDevices.getUserMedia({audio:false,video:cameraConstraints(view.clientWidth,view.clientHeight)});
        if(token!==generation){newStream.getTracks().forEach(t=>t.stop());return false;}
        stream=newStream;video.srcObject=stream;
        stream.getVideoTracks()[0].addEventListener('ended',()=>{if(token===generation)fail('摄像头已断开');});
      }else if(options.source==='file'){
        if(!options.file)throw new Error('未选择视频文件');
        objectURL=URL.createObjectURL(options.file);video.src=objectURL;
      }else video.src=new URL('./test-video.mp4',import.meta.url).href;
      video.loop=true;
      await video.play();
      if(token!==generation)return false;
      inputReady=true;video.style.visibility='visible';$('empty').hidden=true;
      $('run-state').textContent='预览';status('预览中 · 推理未启动');
      return true;
    }catch(error){if(token===generation)fail(inputError(error));return false;}
    finally{if(token===generation)inputPromise=null;}
  });
  return inputPromise;
}

async function start(){
  if(running)return;
  const token=generation;
  resetStats();running=true;runButton(true);updateConfig();
  if(!await ensureInput()||token!==generation||!running)return;
  const mode=options.backend,profile=options.profile;
  $('run-state').textContent='加载中';$('backend-badge').textContent='加载模型';
  try{
    worker=new Worker(new URL('./inference.worker.mjs',import.meta.url),{type:'module'});
    worker.onerror=e=>{if(token===generation)fail(`推理模块加载失败：${e.message}`);};
    worker.onmessage=({data})=>{
      if(token!==generation)return;
      if(data.type==='status')status(data.message);
      if(data.type==='ready'){
        config=data;stats.ready=true;stats.backend=data.backend;
        $('backend-badge').textContent=data.backend==='webgpu'?'WebGPU · GPU':'WASM · CPU';
        $('device').textContent=data.backend==='wasm'?`SegFormer-B0 · CPU / ${data.threads} 线程`:`SegFormer-B0 · ${data.adapterInfo||'GPU'}`;
        $('run-state').textContent='运行中';
        status(data.reason?`已回退 WASM：${data.reason}`:'推理运行中');
        document.body.classList.add('running');schedule();
      }
      if(data.type==='fallback'){
        stats.backend=data.backend;$('backend-badge').textContent='WASM · CPU';$('device').textContent='SegFormer-B0 · CPU';status(`已切换 WASM：${data.reason}`);
      }
      if(data.type==='error')fail(data.message);
      if(data.type==='result'){
        busy=false;if(data.id!==frameId){schedule();return;}
        // Rotation / viewport resize invalidates the crop of an in-flight frame.
        if(capturedViewRevision!==viewRevision){schedule();return;}
        render(data);
        const now=performance.now();
        if(lastResultAt){const value=1000/(now-lastResultAt);fps=fps?fps*.75+value*.25:value;}
        lastResultAt=now;
        Object.assign(stats,{frames:stats.frames+1,backend:data.backend,direction:data.direction,inferenceMs:data.inferenceMs,totalMs:now-capturedAt,fps,floorFraction:data.fraction});
        stats.samples.push({time:now,inferenceMs:data.inferenceMs,totalMs:stats.totalMs});
        if(stats.samples.length>300)stats.samples.shift();
        $('frames').textContent=stats.frames;$('fps').textContent=fps?fps.toFixed(1):'—';$('latency').textContent=Math.round(data.inferenceMs);$('total').textContent=Math.round(stats.totalMs);
        schedule();
      }
    };
    worker.postMessage({type:'init',mode,profile});
  }catch(error){if(token===generation)fail(String(error.message||error));}
}
function schedule(){if(running)timer=setTimeout(tick,0);}
function tick(){
  if(!running||busy||!config)return;
  if(video.readyState<2||video.paused||video.currentTime===lastFrameTime){timer=setTimeout(tick,20);return;}
  lastFrameTime=video.currentTime;capturedAt=performance.now();
  const crop=options.source==='camera'
    ?coverCrop(video.videoWidth,video.videoHeight,view.clientWidth,view.clientHeight)
    :{x:0,y:0,width:video.videoWidth,height:video.videoHeight};
  if(!crop){timer=setTimeout(tick,20);return;}
  const scale=Math.min(1,960/Math.max(crop.width,crop.height));
  const h=Math.round(crop.height*scale),w=Math.round(crop.width*scale);
  if(!w||!h){timer=setTimeout(tick,20);return;}
  capturedViewRevision=viewRevision;
  capture.width=w;capture.height=h;
  captureCtx.drawImage(video,crop.x,crop.y,crop.width,crop.height,0,0,w,h);
  resize.width=config.width;resize.height=config.height;resizeCtx.drawImage(capture,0,0,resize.width,resize.height);
  const pixels=resizeCtx.getImageData(0,0,resize.width,resize.height).data;
  const n=resize.width*resize.height,input=new Float32Array(n*3),mean=[.485,.456,.406],std=[.229,.224,.225];
  for(let i=0;i<n;i++)for(let c=0;c<3;c++)input[c*n+i]=(pixels[i*4+c]/255-mean[c])/std[c];
  busy=true;frameId++;
  worker.postMessage({type:'infer',id:frameId,input,threshold:options.threshold,mode:options.backend},[input.buffer]);
}
function render(data){
  canvas.width=capture.width; canvas.height=capture.height;
  ctx.drawImage(capture,0,0);
  maskCanvas.width=data.width;maskCanvas.height=data.height;
  const image=maskCtx.createImageData(data.width,data.height);
  for(let i=0;i<data.region.length;i++) if(data.region[i]) {image.data[i*4]=66;image.data[i*4+1]=228;image.data[i*4+2]=115;image.data[i*4+3]=85;}
  maskCtx.putImageData(image,0,0);ctx.imageSmoothingEnabled=false;ctx.drawImage(maskCanvas,0,0,canvas.width,canvas.height);ctx.imageSmoothingEnabled=true;
  const sx=canvas.width/data.width,sy=canvas.height/data.height;
  if(data.points.length>1){ctx.beginPath();data.points.forEach(([x,y],i)=>i?ctx.lineTo(x*sx,y*sy):ctx.moveTo(x*sx,y*sy));ctx.strokeStyle='#ffda69';ctx.lineWidth=Math.max(3,canvas.width/150);ctx.lineJoin='round';ctx.stroke();}
  if(data.target){
    const [ax,ay]=data.points[0], [bx,by]=data.target, x1=ax*sx,y1=ay*sy,x2=bx*sx,y2=by*sy;
    const angle=Math.atan2(y2-y1,x2-x1), head=Math.max(15,canvas.width*.045);
    ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.moveTo(x2-head*Math.cos(angle-.55),y2-head*Math.sin(angle-.55));ctx.lineTo(x2,y2);ctx.lineTo(x2-head*Math.cos(angle+.55),y2-head*Math.sin(angle+.55));ctx.strokeStyle='#59c4ff';ctx.lineWidth=Math.max(4,canvas.width/130);ctx.stroke();
  }
  canvas.style.display='block';video.style.visibility='hidden';$('empty').hidden=true;
  const labels={LEFT:['↖','偏左'],RIGHT:['↗','偏右'],FORWARD:['↑','向前'],UNKNOWN:['—','暂无法判断']};
  $('direction').textContent=labels[data.direction][1];
}


function openSettings(){
  $('input-source').value=options.source;$('backend').value=options.backend;$('profile').value=options.profile;
  $('threshold').value=options.threshold;$('threshold-value').textContent=options.threshold.toFixed(2);
  draftFile=options.file;$('file').value='';$('file-name').textContent=draftFile?.name||'未选择文件';
  $('file-field').hidden=options.source!=='file';$('settings-error').textContent='';dialog.showModal();
}
$('start').onclick=()=>running?stop():start();
$('settings').onclick=openSettings;
$('close-settings').onclick=()=>dialog.close();
$('cancel-settings').onclick=()=>dialog.close();
$('input-source').onchange=()=>{$('file-field').hidden=$('input-source').value!=='file';};
$('file').onchange=e=>{draftFile=e.target.files[0]||draftFile;$('file-name').textContent=draftFile?.name||'未选择文件';};
$('threshold').oninput=()=>{$('threshold-value').textContent=Number($('threshold').value).toFixed(2);};
$('settings-form').onsubmit=async event=>{
  event.preventDefault();
  const next={source:$('input-source').value,backend:$('backend').value,profile:$('profile').value,threshold:Number($('threshold').value),file:draftFile};
  if(next.source==='file'&&!next.file){$('settings-error').textContent='请选择视频文件';return;}
  const restart=next.source!==options.source||next.backend!==options.backend||next.profile!==options.profile||(next.source==='file'&&next.file!==options.file);
  const wasRunning=running;
  dialog.close();
  if(restart)stop('配置已更新');
  options=next;updateConfig();
  if(restart){if(wasRunning)await start();else await ensureInput();}
};
video.addEventListener('error',()=>{if(running||inputPromise||inputReady)fail('视频解码失败，请选择浏览器支持的 MP4 / WebM 文件。');});
function invalidateView(){
  viewRevision++;
  if(inputReady){
    canvas.style.display='none';video.style.visibility='visible';
    $('direction').textContent='—';stats.direction='UNKNOWN';lastFrameTime=-1;
  }
}
new ResizeObserver(invalidateView).observe(view);
video.addEventListener('resize',invalidateView);
// Preview also owns camera resources. Release both preview and inference in background.
document.addEventListener('visibilitychange',()=>{if(document.hidden&&(running||inputPromise||inputReady))stop('页面已进入后台，摄像头与推理已停止');});
window.addEventListener('pagehide',()=>stop());
$('compatibility').textContent=!isSecureContext?'非安全连接：摄像头与 WebGPU 需要 HTTPS。':navigator.gpu?'WebGPU API 可用；AUTO 模式在 GPU 不可用时回退 WASM。':'WebGPU API 不可用；AUTO 模式使用 WASM CPU。';
updateConfig();
ensureInput();

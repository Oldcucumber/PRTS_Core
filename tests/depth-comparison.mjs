// Same decoded frames, actual browser workers, and both pipeline outputs.
import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';
process.env.BASE_PATH='/depth-comparison/';
const server=await(await import('../scripts/preview.mjs')).startPreview({port:0});
const base=`http://127.0.0.1:${server.address().port}/depth-comparison/`;
const browser=await chromium.launch({headless:true,channel:'chromium'});
try{
  const page=await browser.newPage();
  await page.route('**/benchmark.html',route=>route.fulfill({contentType:'text/html',body:'<!doctype html><meta charset="utf-8"><title>Depth comparison</title>'}));
  await page.goto(base+'benchmark.html');
  const report=await page.evaluate(async base=>{
    const {containRect}=await import(base+'camera-geometry.mjs');
    const adapter=await navigator.gpu.requestAdapter(),info=adapter.info;
    const video=document.createElement('video');video.muted=true;video.preload='auto';
    video.src=base+'test-video.mp4';
    await new Promise((resolve,reject)=>{video.onloadeddata=resolve;video.onerror=reject;});
    const capture=document.createElement('canvas');
    const scale=Math.min(1,960/Math.max(video.videoWidth,video.videoHeight));
    capture.width=Math.round(video.videoWidth*scale);capture.height=Math.round(video.videoHeight*scale);
    const ctx=capture.getContext('2d'),resize=document.createElement('canvas'),rc=resize.getContext('2d',{willReadFrequently:true});
    const frames=[];
    function prepare(width,height){
      resize.width=width;resize.height=height;
      const fit=containRect(capture.width,capture.height,width,height);
      rc.fillStyle='rgb(124,116,104)';rc.fillRect(0,0,width,height);rc.drawImage(capture,fit.x,fit.y,fit.width,fit.height);
      const pixels=rc.getImageData(0,0,width,height).data,n=width*height,input=new Float32Array(n*3);
      const mean=[.485,.456,.406],std=[.229,.224,.225];
      for(let i=0;i<n;i++)for(let c=0;c<3;c++)input[c*n+i]=(pixels[4*i+c]/255-mean[c])/std[c];
      return {input,contentRect:{x:fit.x/width,y:fit.y/height,width:fit.width/width,height:fit.height/height}};
    }
    for(let i=0;i<10;i++){
      const time=.05+i*.5;
      await new Promise(resolve=>{video.onseeked=resolve;video.currentTime=time;});
      ctx.drawImage(video,0,0,capture.width,capture.height);
      const snapshot=document.createElement('canvas');snapshot.width=capture.width;snapshot.height=capture.height;snapshot.getContext('2d').drawImage(capture,0,0);
      frames.push({time,snapshot,seg:prepare(192,320),depth:prepare(252,252)});
    }
    function request(worker,message){
      return new Promise((resolve,reject)=>{
        const timer=setTimeout(()=>reject(new Error('Worker timeout')),180000);
        worker.onerror=event=>{clearTimeout(timer);reject(new Error(event.message));};
        worker.onmessage=({data})=>{
          if(data.type==='error'){clearTimeout(timer);reject(new Error(data.message));}
          if(data.type==='ready'||data.type==='result'){clearTimeout(timer);resolve(data);}
        };
        worker.postMessage(message);
      });
    }
    const rows=frames.map(frame=>({time:frame.time})),outputs=[];
    for(const enabled of [false,true]){
      const worker=new Worker(base+'inference.worker.mjs?v=depth-1',{type:'module'});
      try{
        await request(worker,{type:'init',mode:'webgpu',profile:'fast',depthEnabled:enabled});
        for(let i=0;i<frames.length;i++){
          const frame=frames[i],t=performance.now();
          const result=await request(worker,{type:'infer',id:i,mode:'webgpu',threshold:.55,input:frame.seg.input,contentRect:frame.seg.contentRect,
            depthInput:enabled?frame.depth.input:undefined,depthContentRect:frame.depth.contentRect});
          const roundTripMs=performance.now()-t;
          if(!enabled){rows[i].segmentation={ms:result.segmentationMs,workerMs:result.totalMs,roundTripMs,direction:result.direction,floorFraction:result.fraction};outputs[i]={baseline:result};}
          else{
            rows[i].fused={segmentationMs:result.segmentationMs,depthMs:result.depthMs,workerMs:result.totalMs,roundTripMs,direction:result.direction,floorFraction:result.fraction,removedFraction:result.depth.removedFraction,fit:result.depth.fit};
            rows[i].sameBaseline=outputs[i].baseline.region.every((v,j)=>v===result.baseline.region[j]);
            rows[i].onlySubtracts=result.region.every((v,j)=>!v||result.baseline.region[j]);
            outputs[i].fused=result;
          }
        }
      }finally{worker.terminate();}
    }
    const sheet=document.createElement('canvas'),tileWidth=216,tileHeight=384,header=48;
    sheet.width=tileWidth*4;sheet.height=(tileHeight+header)*5;
    const sc=sheet.getContext('2d'),mask=document.createElement('canvas'),mc=mask.getContext('2d');
    sc.fillStyle='#111820';sc.fillRect(0,0,sheet.width,sheet.height);
    function draw(result,frame,x,y){
      sc.drawImage(frame.snapshot,x,y,tileWidth,tileHeight);
      mask.width=result.width;mask.height=result.height;
      const pixels=mc.createImageData(mask.width,mask.height);
      for(let i=0;i<result.region.length;i++){
        if(result.depth?.excluded[i])pixels.data.set([255,125,45,190],i*4);
        else if(result.region[i])pixels.data.set([66,228,115,85],i*4);
      }
      mc.putImageData(pixels,0,0);sc.imageSmoothingEnabled=false;sc.drawImage(mask,x,y,tileWidth,tileHeight);sc.imageSmoothingEnabled=true;
      sc.beginPath();result.points.forEach(([px,py],i)=>{const xx=x+px/result.width*tileWidth,yy=y+py/result.height*tileHeight;i?sc.lineTo(xx,yy):sc.moveTo(xx,yy);});
      sc.strokeStyle='#ffda69';sc.lineWidth=2;sc.stroke();
    }
    frames.forEach((frame,i)=>{
      const x=i%2*tileWidth*2,y=Math.floor(i/2)*(tileHeight+header);
      sc.fillStyle='#ffffff';sc.font='13px sans-serif';
      sc.fillText(`${frame.time.toFixed(2)}s | Segmentation`,x+6,y+18);
      sc.fillText(`+ Depth | excluded ${(rows[i].fused.removedFraction*100).toFixed(2)}%`,x+tileWidth+6,y+18);
      sc.fillStyle='#a9b7c8';sc.fillText(rows[i].segmentation.direction,x+6,y+38);sc.fillText(rows[i].fused.direction,x+tileWidth+6,y+38);
      draw(outputs[i].baseline,frame,x,y+header);draw(outputs[i].fused,frame,x+tileWidth,y+header);
    });
    return {device:[info.vendor,info.architecture,info.description].filter(Boolean).join(' / '),userAgent:navigator.userAgent,profile:'fast',depthInput:'252x252',threshold:.55,rows,sheet:sheet.toDataURL('image/png')};
  },base);
  await mkdir('outputs/depth',{recursive:true});
  await writeFile('outputs/depth/fixed-frame-comparison.png',Buffer.from(report.sheet.split(',')[1],'base64'));delete report.sheet;
  for(const row of report.rows){assert.ok(row.sameBaseline);assert.ok(row.onlySubtracts);}
  const mean=values=>values.reduce((a,b)=>a+b,0)/values.length;
  const steady=report.rows.slice(1);
  report.summary={frames:report.rows.length,validFits:report.rows.filter(r=>r.fused.fit.valid).length,
    changedDirections:report.rows.filter(r=>r.fused.direction!==r.segmentation.direction).length,
    meanExcludedFraction:mean(report.rows.map(r=>r.fused.removedFraction)),maxExcludedFraction:Math.max(...report.rows.map(r=>r.fused.removedFraction)),
    segmentationMs:mean(steady.map(r=>r.segmentation.ms)),depthMs:mean(steady.map(r=>r.fused.depthMs)),
    fusedWorkerMs:mean(steady.map(r=>r.fused.workerMs)),timingNote:'First measured frame excluded; sequential fixed-frame worker timing excludes capture, preprocessing and rendering. No annotated ground truth.'};
  await writeFile('outputs/depth/fixed-frame-comparison.json',JSON.stringify(report,null,2));
  console.log(JSON.stringify({device:report.device,summary:report.summary},null,2));
}finally{await browser.close();await new Promise(resolve=>server.close(resolve));}

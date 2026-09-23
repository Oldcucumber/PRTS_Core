import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';

process.env.BASE_PATH='/camera-test/';
const remote=process.env.TEST_BASE_URL;
const server=remote?null:await(await import('../scripts/preview.mjs')).startPreview({port:0});
const base=remote||`http://127.0.0.1:${server.address().port}/camera-test/`;
const browser=await chromium.launch({headless:true,channel:'chromium'});
const context=await browser.newContext({viewport:{width:390,height:844}});
const page=await context.newPage(),errors=[];
page.on('pageerror',error=>errors.push(error.message));
await page.addInitScript(()=>{
  const sensor=document.createElement('canvas');sensor.width=1280;sensor.height=720;
  const ctx=sensor.getContext('2d');
  function paint(){
    const w=sensor.width,h=sensor.height;
    ctx.fillStyle='#cc2424';ctx.fillRect(0,0,w/3,h);
    ctx.fillStyle='#28783c';ctx.fillRect(w/3,0,w/3,h);
    ctx.fillStyle='#244ccc';ctx.fillRect(w*2/3,0,w/3,h);
    ctx.fillStyle='#8050d0';ctx.fillRect(0,0,w,h/9);
    ctx.fillStyle='#e0c030';ctx.fillRect(0,h*8/9,w,h/9);
  }
  paint();setInterval(paint,33);
  window.rotateSensor=()=>{sensor.width=720;sensor.height=1280;paint();};
  Object.defineProperty(navigator.mediaDevices,'getUserMedia',{value:async constraints=>{
    window.requestedCameraConstraints=constraints;return sensor.captureStream(30);
  }});
  const original=Worker.prototype.postMessage;
  Worker.prototype.postMessage=function(message,...args){
    if(message.type==='infer'){
      const rect=message.contentRect,w=192,h=320,n=w*h;
      function pixel(rx,ry){const x=Math.floor((rect.x+rect.width*rx)*w),y=Math.floor((rect.y+rect.height*ry)*h),i=y*w+x;
        return [0,1,2].map(c=>(message.input[c*n+i]*[.229,.224,.225][c]+[.485,.456,.406][c])*255);}
      window.lastModelInput={rect,left:pixel(.05,.5),right:pixel(.95,.5),top:pixel(.5,.025),bottom:pixel(.5,.975)};
    }
    return original.call(this,message,...args);
  };
});
await mkdir('outputs/camera',{recursive:true});
function checkBorders(p){
  assert.ok(p.left[0]>p.left[1]*1.5&&p.left[0]>p.left[2]*1.5,`left edge ${p.left}`);
  assert.ok(p.right[2]>p.right[0]*1.5&&p.right[2]>p.right[1]*1.5,`right edge ${p.right}`);
  assert.ok(p.top[2]>p.top[0]);
  assert.ok(p.bottom[0]>p.bottom[2]&&p.bottom[1]>p.bottom[2]);
}
async function frameBorders(){return page.locator('#result').evaluate(c=>{
  const ctx=c.getContext('2d'),pixel=(x,y)=>Array.from(ctx.getImageData(Math.floor(c.width*x),Math.floor(c.height*y),1,1).data);
  return {width:c.width,height:c.height,left:pixel(.05,.5),right:pixel(.95,.5),top:pixel(.5,.025),bottom:pixel(.5,.975)};
});}
try{
  await page.goto(base,{waitUntil:'networkidle'});await page.click('#use-camera');
  await page.waitForFunction(()=>document.querySelector('video').videoWidth===1280&&document.getElementById('run-state').textContent==='预览');
  assert.equal(await page.locator('#source').evaluate(v=>getComputedStyle(v).objectFit),'contain');
  const requested=await page.evaluate(()=>window.requestedCameraConstraints.video);
  assert.equal(requested.resizeMode.ideal,'none');assert.equal('aspectRatio' in requested,false);
  await page.locator('.camera-view').scrollIntoViewIfNeeded();
  const screenshot=await page.screenshot({path:'outputs/camera/full-frame-preview.png'});
  const preview=await page.evaluate(async bytes=>{
    const bitmap=await createImageBitmap(new Blob([new Uint8Array(bytes)],{type:'image/png'}));
    const c=document.createElement('canvas');c.width=bitmap.width;c.height=bitmap.height;
    const ctx=c.getContext('2d');ctx.drawImage(bitmap,0,0);
    // The original red/blue borders must remain visible on both sides.
    const pixel=(x,y)=>Array.from(ctx.getImageData(x,y,1,1).data);
    const v=document.querySelector('video'),r=v.getBoundingClientRect(),scale=Math.min(r.width/v.videoWidth,r.height/v.videoHeight),w=v.videoWidth*scale,h=v.videoHeight*scale,x=r.x+(r.width-w)/2,y=r.y+(r.height-h)/2;
    return {left:pixel(Math.round(x+w*.05),Math.round(y+h*.5)),right:pixel(Math.round(x+w*.95),Math.round(y+h*.5))};
  },Array.from(screenshot));
  assert.ok(preview.left[0]>preview.left[1]*2);assert.ok(preview.right[2]>preview.right[0]*2);
  await page.click('#settings');await page.selectOption('#backend','wasm');await page.click('.apply');await page.click('#start');
  await page.waitForFunction(()=>window.floorLabStats.frames>=3||window.floorLabStats.errors.length,null,{timeout:120000});
  assert.deepEqual(await page.evaluate(()=>window.floorLabStats.errors),[]);
  const input=await page.evaluate(()=>window.lastModelInput);checkBorders(input);
  assert.ok(Math.abs((input.rect.width*192)/(input.rect.height*320)-1280/720)<.01);
  const portraitView=await frameBorders();checkBorders(portraitView);
  assert.equal(portraitView.width,960);assert.equal(portraitView.height,540);
  await page.screenshot({path:'outputs/camera/full-frame-inference.png'});
  const previous=await page.evaluate(()=>window.floorLabStats.frames);
  await page.setViewportSize({width:844,height:390});
  await page.waitForFunction(n=>window.floorLabStats.frames>n+2,previous);
  const landscapeView=await frameBorders();checkBorders(landscapeView);
  assert.equal(landscapeView.width,960);assert.equal(landscapeView.height,540);
  // Real orientation change of the video source, not just CSS viewport.
  await page.evaluate(()=>window.rotateSensor());
  await page.setViewportSize({width:390,height:844});
  await page.waitForFunction(()=>{const c=document.getElementById('result');return getComputedStyle(c).display!=='none'&&c.width===540&&c.height===960;},null,{timeout:15000});
  const rotatedSource=await frameBorders();checkBorders(rotatedSource);
  checkBorders(await page.evaluate(()=>window.lastModelInput));
  await page.click('#start');
  assert.deepEqual(errors,[]);
  await writeFile(`outputs/camera/${remote?'live-':''}validation.json`,JSON.stringify({base,requested,preview,input,portraitView,landscapeView,rotatedSource,errors},null,2));
  console.log('Full-frame checks passed: all four edges in preview, model input and overlay; aspect preserved; viewport and sensor rotation.');
}finally{await browser.close();if(server)await new Promise(resolve=>server.close(resolve));}

import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';

process.env.BASE_PATH='/camera-test/';
const remote=process.env.TEST_BASE_URL;
const server=remote?null:await(await import('../scripts/preview.mjs')).startPreview({port:0});
const base=remote||`http://127.0.0.1:${server.address().port}/camera-test/`;
const browser=await chromium.launch({headless:true,channel:'chromium'});
const context=await browser.newContext({viewport:{width:390,height:844}});
const page=await context.newPage();
const errors=[];
page.on('pageerror',error=>errors.push(error.message));
// Reproduce a phone returning landscape sensor frames despite portrait ideals.
await page.addInitScript(()=>{
  const sensor=document.createElement('canvas');sensor.width=1280;sensor.height=720;
  const ctx=sensor.getContext('2d');
  function paint(){
    ctx.fillStyle='#cc2424';ctx.fillRect(0,0,430,720);
    ctx.fillStyle='#28783c';ctx.fillRect(430,0,420,720);
    ctx.fillStyle='#244ccc';ctx.fillRect(850,0,430,720);
    ctx.fillStyle='#8050d0';ctx.fillRect(0,0,1280,80);
    ctx.fillStyle='#e0c030';ctx.fillRect(0,640,1280,80);
    ctx.strokeStyle='#ffffff';ctx.lineWidth=4;
    ctx.beginPath();ctx.moveTo(600,360);ctx.lineTo(680,360);ctx.moveTo(640,320);ctx.lineTo(640,400);ctx.stroke();
  }
  paint();setInterval(paint,33);
  Object.defineProperty(navigator.mediaDevices,'getUserMedia',{value:async constraints=>{
    window.requestedCameraConstraints=constraints;
    return sensor.captureStream(30);
  }});
});
await mkdir('outputs/camera',{recursive:true});
try{
  await page.goto(base,{waitUntil:'networkidle'});
  await page.waitForFunction(()=>document.querySelector('video').videoWidth===1280&&document.getElementById('run-state').textContent==='预览');
  assert.equal(await page.locator('#source').evaluate(v=>getComputedStyle(v).objectFit),'cover');
  const requested=await page.evaluate(()=>window.requestedCameraConstraints.video);
  assert.ok(requested.width.ideal<requested.height.ideal);
  const preview=await page.screenshot({path:'outputs/camera/portrait-preview.png'});
  const pixels=await page.evaluate(async bytes=>{
    const bitmap=await createImageBitmap(new Blob([new Uint8Array(bytes)],{type:'image/png'}));
    const c=document.createElement('canvas');c.width=bitmap.width;c.height=bitmap.height;
    const ctx=c.getContext('2d');ctx.drawImage(bitmap,0,0);
    return [[6,240],[6,740]].map(([x,y])=>Array.from(ctx.getImageData(x,y,1,1).data));
  },Array.from(preview));
  // These pixels were black above/below the letterboxed stream before this fix.
  for(const [r,g,b] of pixels)assert.ok(g>r*1.5&&g>b*1.5,`preview crop pixel ${[r,g,b]}`);
  await page.click('#settings');await page.selectOption('#backend','wasm');await page.click('.apply');
  await page.click('#start');
  await page.waitForFunction(()=>window.floorLabStats.frames>=3||window.floorLabStats.errors.length,null,{timeout:120000});
  assert.deepEqual(await page.evaluate(()=>window.floorLabStats.errors),[]);
  const portrait=await page.locator('#result').evaluate(c=>{
    const ctx=c.getContext('2d');
    return {width:c.width,height:c.height,
      top:Array.from(ctx.getImageData(Math.floor(c.width*.1),Math.floor(c.height*.025),1,1).data),
      bottom:Array.from(ctx.getImageData(Math.floor(c.width*.1),Math.floor(c.height*.975),1,1).data)};
  });
  assert.ok(Math.abs(portrait.width/portrait.height-390/844)<.002);
  assert.ok(portrait.top[2]>portrait.top[0]);
  assert.ok(portrait.bottom[0]>portrait.bottom[2]&&portrait.bottom[1]>portrait.bottom[2]);
  await page.screenshot({path:'outputs/camera/portrait-inference.png'});
  await page.setViewportSize({width:844,height:390});
  await page.waitForFunction(()=>{
    const c=document.getElementById('result');
    return getComputedStyle(c).display!=='none'&&Math.abs(c.width/c.height-844/390)<.003;
  },null,{timeout:15000});
  const landscape=await page.locator('#result').evaluate(c=>({width:c.width,height:c.height}));
  await page.screenshot({path:'outputs/camera/landscape-inference.png'});
  await page.setViewportSize({width:390,height:760});
  await page.waitForFunction(()=>{
    const c=document.getElementById('result');
    return getComputedStyle(c).display!=='none'&&Math.abs(c.width/c.height-390/760)<.002;
  },null,{timeout:15000});
  const resized=await page.locator('#result').evaluate(c=>({width:c.width,height:c.height}));
  await page.click('#start');
  await page.click('#settings');await page.selectOption('#input-source','sample');await page.click('.apply');
  await page.waitForFunction(()=>document.getElementById('run-state').textContent==='预览');
  assert.equal(await page.locator('#source').evaluate(v=>getComputedStyle(v).objectFit),'contain');
  assert.deepEqual(errors,[]);
  await writeFile(`outputs/camera/${remote?'live-':''}validation.json`,JSON.stringify({base,requested,previewPixels:pixels,portrait,landscape,resized,errors},null,2));
  console.log('Camera checks passed: landscape sensor -> portrait cover, upright inference crop, rotation, viewport resize, uncropped file video.');
}finally{
  await browser.close();
  if(server)await new Promise(resolve=>server.close(resolve));
}

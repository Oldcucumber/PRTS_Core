import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';

process.env.BASE_PATH='/camera-selection/';
const remote=process.env.TEST_BASE_URL;
const server=remote?null:await(await import('../scripts/preview.mjs')).startPreview({port:0});
const base=remote||`http://127.0.0.1:${server.address().port}/camera-selection/`;
const browser=await chromium.launch({headless:true,channel:'chromium'});
const context=await browser.newContext({viewport:{width:390,height:844}});
const page=await context.newPage();
const errors=[];page.on('pageerror',error=>errors.push(error.message));
await page.addInitScript(()=>{
  window.testCameras=[
    {kind:'videoinput',deviceId:'tele',label:'后置长焦镜头'},
    {kind:'videoinput',deviceId:'wide',label:'后置主摄广角'},
    {kind:'videoinput',deviceId:'front',label:'前置摄像头'},
  ];
  window.cameraCalls=[];window.cameraTracks=[];window.enumerationFails=false;
  const sensor=document.createElement('canvas');sensor.width=720;sensor.height=1280;
  const ctx=sensor.getContext('2d');
  function paint(){ctx.fillStyle='#284354';ctx.fillRect(0,0,720,1280);ctx.fillStyle='#bbc9c0';ctx.fillRect(0,600,720,680);}
  paint();setInterval(paint,33);
  Object.defineProperty(navigator.mediaDevices,'enumerateDevices',{value:async()=>{
    if(window.enumerationFails)throw new Error('Device enumeration unavailable');
    return window.testCameras;
  }});
  Object.defineProperty(navigator.mediaDevices,'getUserMedia',{value:async constraints=>{
    window.cameraCalls.push({constraints,oldLive:window.cameraTracks.filter(t=>t.readyState==='live').length});
    const id=constraints.video.deviceId?.exact||'tele';
    const device=window.testCameras.find(d=>d.deviceId===id);
    if(!device)throw new DOMException('Device not available','OverconstrainedError');
    const stream=sensor.captureStream(30),track=stream.getVideoTracks()[0];
    Object.defineProperty(track,'label',{value:device.label});
    Object.defineProperty(track,'getSettings',{value:()=>({deviceId:id,width:720,height:1280})});
    window.cameraTracks.push(track);return stream;
  }});
});
async function preview(id){await page.waitForFunction(id=>{
  const v=document.querySelector('video');
  return v.srcObject?.getVideoTracks()[0].getSettings().deviceId===id&&document.getElementById('run-state').textContent==='预览';
},id);}
async function select(id,backend){
  await page.click('#settings');
  await page.waitForFunction(()=>!document.getElementById('refresh-cameras').disabled);
  await page.selectOption('#camera-device',id);
  if(backend)await page.selectOption('#backend',backend);
  await page.click('.apply');
}
async function inference(id){
  await page.waitForFunction(id=>window.floorLabStats.frames>=3&&document.querySelector('video').srcObject?.getVideoTracks()[0].getSettings().deviceId===id,id,{timeout:120000});
  assert.deepEqual(await page.evaluate(()=>window.floorLabStats.errors),[]);
}
try{
  await page.goto(base,{waitUntil:'networkidle'});await page.waitForFunction(()=>window.prtsSession);await page.evaluate(()=>{document.getElementById('visual-only').checked=true;prtsSession.speech.setMuted(true);return prtsControls.source('camera');});await preview('tele');
  assert.equal(await page.locator('#source-label').textContent(),'后置长焦镜头');
  await page.click('#settings');await page.waitForFunction(()=>document.getElementById('camera-device').options.length===4);
  await page.selectOption('#camera-device','wide');
  await page.click('#refresh-cameras');await page.waitForFunction(()=>!document.getElementById('refresh-cameras').disabled);
  assert.equal(await page.locator('#camera-device').inputValue(),'wide');
  await mkdir('outputs/camera-selection',{recursive:true});
  await page.screenshot({path:'outputs/camera-selection/settings.png'});
  await page.click('#cancel-settings');await preview('tele');
  assert.equal(await page.evaluate(()=>localStorage.getItem('floor-lab.camera-device')),null);
  await select('wide');await preview('wide');
  assert.equal(await page.locator('#source-label').textContent(),'后置主摄广角');
  assert.equal(await page.evaluate(()=>window.cameraTracks[0].readyState),'ended');
  assert.equal(await page.evaluate(()=>localStorage.getItem('floor-lab.camera-device')),'wide');
  const constraints=await page.evaluate(()=>window.cameraCalls.at(-1).constraints.video);
  assert.deepEqual(constraints.deviceId,{exact:'wide'});assert.equal('facingMode' in constraints,false);
  await page.reload({waitUntil:'networkidle'});await page.waitForFunction(()=>window.prtsSession);await page.evaluate(()=>{document.getElementById('visual-only').checked=true;prtsSession.speech.setMuted(true);return prtsControls.source('camera');});await preview('wide');
  assert.equal(await page.evaluate(()=>window.cameraCalls[0].constraints.video.deviceId.exact),'wide');
  await select('wide','wasm');await preview('wide');await page.click('#start');await inference('wide');
  await select('front');await inference('front');
  assert.equal(await page.locator('#start-label').textContent(),'结束');
  assert.equal(await page.locator('#source-label').textContent(),'前置摄像头');
  assert.ok(await page.evaluate(()=>window.cameraCalls.every(c=>c.oldLive===0)));
  await page.click('#start');
  // Lost device stays selected until the user chooses a replacement.
  await page.evaluate(()=>{window.testCameras=window.testCameras.filter(d=>d.deviceId!=='front');navigator.mediaDevices.dispatchEvent(new Event('devicechange'));});
  await page.click('#settings');await page.waitForFunction(()=>!document.getElementById('refresh-cameras').disabled);
  assert.equal(await page.locator('#camera-device').inputValue(),'front');
  assert.match(await page.locator('#camera-device option:checked').textContent(),/未出现在列表/);
  await page.click('#cancel-settings');await page.click('#start');
  await page.waitForFunction(()=>window.floorLabStats.errors.length>0);
  assert.match(await page.locator('#status').textContent(),/所选摄像头不可用/);
  await select('');await preview('tele');
  assert.equal(await page.evaluate(()=>localStorage.getItem('floor-lab.camera-device')),null);
  await page.evaluate(()=>{window.enumerationFails=true;});
  await page.click('#settings');await page.waitForFunction(()=>document.getElementById('camera-help').textContent.includes('无法刷新'));
  // Enumerating devices is optional: a failure must not stop the live preview.
  assert.equal(await page.locator('#run-state').textContent(),'预览');
  await page.click('#close-settings');
  assert.deepEqual(errors,[]);
  await writeFile(`outputs/camera-selection/${remote?'live-':''}validation.json`,JSON.stringify({base,passed:['enumeration','refresh preserves selection','cancel','exact lens selection','old track release','remember on reload','switch during inference','device removal','auto recovery','enumeration failure'],constraints,errors},null,2));
  console.log('Camera selection passed: exact device, release, persistence, inference restart, missing-device recovery.');
}finally{await browser.close();if(server)await new Promise(resolve=>server.close(resolve));}

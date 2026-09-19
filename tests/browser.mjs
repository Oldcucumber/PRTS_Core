import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';

await mkdir('outputs/web',{recursive:true});
const browser=await chromium.launch({headless:true,channel:'chromium',args:['--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream']});
const context=await browser.newContext({viewport:{width:1280,height:900},permissions:['camera']});
const page=await context.newPage();
const errors=[];
page.on('pageerror',e=>{errors.push(e.message);console.log('PAGE ERROR',e.message);});
const results={};
async function waitFrames(n=8){
  await page.waitForFunction(n=>window.floorLabStats.frames>=n||window.floorLabStats.errors.length,n,{timeout:120000});
  const stats=await page.evaluate(()=>({...window.floorLabStats}));
  assert.deepEqual(stats.errors,[],await page.locator('#status').textContent());
  assert.ok(stats.frames>=n);delete stats.samples;return stats;
}
async function configure({source,backend,profile,file}={}){
  await page.click('#settings');
  assert.equal(await page.locator('#settings-dialog').evaluate(d=>d.open),true);
  if(source)await page.selectOption('#input-source',source);
  if(backend)await page.selectOption('#backend',backend);
  if(profile)await page.selectOption('#profile',profile);
  if(file)await page.setInputFiles('#file',file);
  await page.click('.apply');
  await page.waitForFunction(()=>!document.getElementById('settings-dialog').open);
}
async function stop(){await page.click('#start');assert.equal(await page.locator('#start-label').textContent(),'开始');}
try{
  await page.goto('http://localhost:8080');
  await page.waitForFunction(()=>document.querySelector('video').srcObject?.active);
  assert.equal(await page.locator('main button:visible').count(),2);
  assert.equal(await page.locator('#start-label').textContent(),'开始');
  assert.equal(await page.evaluate(()=>window.floorLabStats.frames),0);
  assert.equal(await page.locator('#settings-dialog').isVisible(),false);
  assert.doesNotMatch(await page.locator('body').innerText(),/看见地面|找到方向|地面识别 \/ 方向估计|FLOOR LAB/);
  await page.screenshot({path:'outputs/web/camera-preview.png'});
  // Cancel and Escape leave live configuration unchanged.
  await page.click('#settings');await page.selectOption('#profile','quality');
  await page.click('#cancel-settings');assert.match(await page.locator('#config-summary').textContent(),/192 × 320/);
  await page.click('#settings');await page.keyboard.press('Escape');
  assert.equal(await page.locator('#settings-dialog').isVisible(),false);
  for(const backend of ['wasm','webgpu']){
    await configure({source:'sample',backend});await page.click('#start');
    results[backend]=await waitFrames(12);
    assert.equal(results[backend].backend,backend);
    assert.ok(results[backend].floorFraction>.05);
    await page.screenshot({path:`outputs/web/${backend}.png`});
    // Settings remains available during inference; a cancelled edit is not applied.
    await page.click('#settings');await page.selectOption('#profile','quality');await page.click('#cancel-settings');
    assert.match(await page.locator('#config-summary').textContent(),/192 × 320/);
    await stop();const count=await page.evaluate(()=>window.floorLabStats.frames);
    await page.waitForTimeout(200);assert.equal(await page.evaluate(()=>window.floorLabStats.frames),count);
  }
  await configure({profile:'quality'});await page.click('#start');results.quality=await waitFrames(8);
  // Apply while running restarts with the new configuration.
  await configure({profile:'fast'});await waitFrames(6);assert.match(await page.locator('#config-summary').textContent(),/192 × 320/);await stop();
  await configure({source:'file',backend:'wasm',file:'VID20260919182406.mp4'});
  await page.click('#start');results.file=await waitFrames(3);await stop();
  await configure({source:'camera'});
  await page.waitForFunction(()=>document.querySelector('video').srcObject?.active);
  await page.click('#start');results.camera=await waitFrames(3);
  await page.evaluate(()=>{window.testTrack=document.querySelector('video').srcObject.getVideoTracks()[0];});
  await stop();assert.equal(await page.evaluate(()=>window.testTrack.readyState),'ended');
  await page.route('**/inference.worker.mjs',async route=>{
    const response=await route.fetch();
    await route.fulfill({response,body:`Object.defineProperty(navigator,'gpu',{value:undefined});\n${await response.text()}`});
  });
  await configure({source:'sample',backend:'auto'});await page.click('#start');results.fallback=await waitFrames(3);
  assert.equal(results.fallback.backend,'wasm');assert.match(await page.locator('#status').textContent(),/回退/);await stop();
  await configure({backend:'webgpu'});await page.click('#start');
  await page.waitForFunction(()=>window.floorLabStats.errors.length>0,null,{timeout:30000});
  assert.match(await page.locator('#status').textContent(),/WebGPU 启动失败/);
  assert.equal(await page.locator('#start-label').textContent(),'开始');
  await page.unroute('**/inference.worker.mjs');
  await page.setViewportSize({width:390,height:844});
  await configure({backend:'auto'});await page.click('#start');results.mobile=await waitFrames(6);
  await page.screenshot({path:'outputs/web/mobile.png'});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth&&document.documentElement.scrollHeight<=innerHeight),true);
  await page.click('#settings');await page.screenshot({path:'outputs/web/settings-mobile.png'});
  assert.equal(await page.locator('#settings-dialog').evaluate(d=>{const r=d.getBoundingClientRect();return r.top>=0&&r.bottom<=innerHeight&&r.left>=0&&r.right<=innerWidth;}),true);
  await page.click('#close-settings');await stop();
  // Permission-denied startup is recoverable using the settings dialog.
  const denied=await browser.newContext({permissions:[]});
  const deniedPage=await denied.newPage();
  await deniedPage.addInitScript(()=>Object.defineProperty(navigator.mediaDevices,'getUserMedia',{value:async()=>{throw new DOMException('Denied','NotAllowedError');}}));
  await deniedPage.goto('http://localhost:8080');
  await deniedPage.waitForFunction(()=>document.getElementById('run-state').textContent==='错误');
  assert.match(await deniedPage.locator('#status').textContent(),/权限被拒绝/);
  await deniedPage.click('#settings');await deniedPage.selectOption('#input-source','sample');await deniedPage.click('.apply');
  await deniedPage.waitForFunction(()=>document.getElementById('run-state').textContent==='预览');
  await denied.close();
  assert.deepEqual(errors,[]);
  await writeFile('outputs/web/ui-validation.json',JSON.stringify(results,null,2));
  console.log('Browser UI checks passed: camera preview, two controls, modal apply/cancel, inference, fallbacks, camera release, mobile layout, permission recovery');
}finally{await browser.close();}

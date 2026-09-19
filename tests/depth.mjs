import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';
process.env.BASE_PATH='/depth-test/';
const remote=process.env.TEST_BASE_URL;
const server=remote?null:await(await import('../scripts/preview.mjs')).startPreview({port:0});
const base=remote||`http://127.0.0.1:${server.address().port}/depth-test/`;
const browser=await chromium.launch({headless:true,channel:'chromium',args:['--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream']});
const context=await browser.newContext({permissions:['camera'],viewport:{width:1280,height:900}}),page=await context.newPage();
const errors=[],requests=[];page.on('pageerror',e=>errors.push(e.message));context.on('request',r=>requests.push(r.url()));
page.on('console',msg=>{if(msg.type()==='error')console.log(msg.text().slice(0,400));});
const report={base};
async function configure(enabled,view='overlay',backend='webgpu'){
  await page.click('#settings');await page.selectOption('#input-source','sample');await page.selectOption('#backend',backend);
  await page.locator('#depth-enabled').setChecked(enabled);
  if(enabled)await page.selectOption('#depth-view',view);
  await page.click('.apply');
}
async function ready(enabled,frames=8){
  await page.waitForFunction(({enabled,frames})=>window.floorLabStats.errors.length||window.floorLabStats.frames>=frames&&window.floorLabStats.depthEnabled===enabled,{enabled,frames},{timeout:180000});
  const result=await page.evaluate(()=>({...window.floorLabStats}));delete result.samples;
  assert.deepEqual(result.errors,[]);assert.equal(result.depthEnabled,enabled);return result;
}
try{
  await page.goto(base);await page.waitForFunction(()=>document.querySelector('video').srcObject?.active);
  await configure(false);await page.click('#start');report.segmentation=await ready(false);
  assert.equal(requests.some(url=>url.includes('depth-small.onnx')),false);
  await configure(true,'compare');report.fused=await ready(true,15);
  assert.equal(report.fused.depthBackend,'webgpu');assert.ok(report.fused.depthMs>0);
  assert.ok(report.fused.floorFraction<=report.fused.baselineFloorFraction);
  assert.equal(await page.locator('#result').evaluate(c=>c.width/c.height),1080/1920*2);
  await mkdir('outputs/depth',{recursive:true});
  await page.screenshot({path:'outputs/depth/comparison.png'});
  await page.setViewportSize({width:390,height:844});
  const beforeResize=await page.evaluate(()=>window.floorLabStats.frames);
  await page.waitForFunction(frames=>window.floorLabStats.frames>frames+2,beforeResize);
  await page.screenshot({path:'outputs/depth/mobile-comparison.png'});
  await page.click('#settings');await page.screenshot({path:'outputs/depth/mobile-settings.png'});await page.click('#cancel-settings');
  await page.setViewportSize({width:1280,height:900});
  await configure(true,'depth');await ready(true,18);
  await page.waitForFunction(()=>document.querySelector('#result').width/document.querySelector('#result').height===1080/1920);
  await page.screenshot({path:'outputs/depth/heatmap.png'});
  await configure(false);report.disabled=await ready(false);assert.equal(report.disabled.depthMs,0);
  await configure(true,'overlay','wasm');report.wasm=await ready(true,3);assert.equal(report.wasm.depthBackend,'wasm');
  await page.click('#start');
  // Missing depth assets must not be reported as a successful fused run.
  await page.route('**/depth-small.onnx',route=>route.fulfill({status:503,body:'Unavailable'}));
  await page.click('#start');await page.waitForFunction(()=>window.floorLabStats.errors.length>0,null,{timeout:30000});
  assert.match(await page.locator('#status').textContent(),/深度模型加载失败/);
  await configure(false);await page.click('#start');report.recovery=await ready(false,3);
  await page.click('#start');
  assert.deepEqual(errors,[]);
  await writeFile(`outputs/depth/${remote?'live-':''}ui-validation.json`,JSON.stringify(report,null,2));
  console.log(JSON.stringify(report,null,2));console.log('Depth UI checks passed');
}finally{await browser.close();if(server)await new Promise(resolve=>server.close(resolve));}

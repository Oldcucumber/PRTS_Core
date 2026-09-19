import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';

// An arbitrary nested repository path catches accidental root-relative URLs.
process.env.BASE_PATH='/pages-test/PRTS_Core/';
const {startPreview}=await import('../scripts/preview.mjs');
const server=await startPreview({port:0});
const base=`http://127.0.0.1:${server.address().port}/pages-test/PRTS_Core/`;
const browser=await chromium.launch({headless:true,channel:'chromium',args:['--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream']});
const errors=[],requests=new Set(),failures=[];
const results={basePath:'/pages-test/PRTS_Core/',crossOriginIsolated:null,cases:{}};
try{
  const manifest=JSON.parse(await readFile('dist/asset-manifest.json','utf8'));
  for(const asset of manifest.files){
    const data=await readFile(`dist/${asset.path}`);
    assert.equal(data.length,asset.bytes);
    assert.equal(createHash('sha256').update(data).digest('hex'),asset.sha256);
  }
  const context=await browser.newContext({permissions:['camera'],viewport:{width:390,height:844}});
  await context.route('**/*',route=>{
    const url=route.request().url();
    if(/^https?:/.test(url)&&!url.startsWith(base)){
      failures.push(`Request outside static site: ${url}`);
      return route.abort();
    }
    return route.continue();
  });
  context.on('request',request=>{if(/^https?:/.test(request.url()))requests.add(request.url().replace(base,''));});
  context.on('response',response=>{if(response.status()>=400)failures.push(`${response.status()}: ${response.url()}`);});
  const page=await context.newPage();
  page.on('pageerror',error=>errors.push(error.message));
  const response=await page.goto(base);
  assert.equal(response.headers()['cross-origin-opener-policy'],undefined);
  assert.equal(response.headers()['cross-origin-embedder-policy'],undefined);
  results.crossOriginIsolated=await page.evaluate(()=>crossOriginIsolated);
  assert.equal(results.crossOriginIsolated,false);
  await page.waitForFunction(()=>document.querySelector('video').srcObject?.active);
  async function configure(backend,profile='fast',source='sample'){
    await page.click('#settings');
    await page.selectOption('#input-source',source);
    await page.selectOption('#backend',backend);
    await page.selectOption('#profile',profile);
    await page.click('.apply');
  }
  async function run(name,backend){
    await page.click('#start');
    await page.waitForFunction(()=>window.floorLabStats.frames>=8||window.floorLabStats.errors.length,null,{timeout:120000});
    const stats=await page.evaluate(()=>({...window.floorLabStats}));delete stats.samples;
    assert.deepEqual(stats.errors,[]);
    assert.ok(stats.frames>=8);
    assert.equal(stats.backend,backend);
    results.cases[name]={...stats,device:await page.locator('#device').textContent()};
    await page.click('#start');
  }
  await configure('wasm');await run('wasmSingleThread','wasm');
  assert.match(results.cases.wasmSingleThread.device,/1 线程/);
  await configure('webgpu');await run('webgpuFast','webgpu');
  await configure('webgpu','quality');await run('webgpuQuality','webgpu');
  await page.route('**/inference.worker.mjs*',async route=>{
    const response=await route.fetch();
    await route.fulfill({response,body:`Object.defineProperty(navigator,'gpu',{value:undefined});\n${await response.text()}`});
  });
  await configure('auto');await run('autoFallback','wasm');
  await page.unroute('**/inference.worker.mjs*');
  await configure('auto','fast','camera');
  await page.waitForFunction(()=>document.querySelector('video').srcObject?.active);
  await page.evaluate(()=>{window.trackForTest=document.querySelector('video').srcObject.getVideoTracks()[0];});
  await run('camera','webgpu');
  assert.equal(await page.evaluate(()=>window.trackForTest.readyState),'ended');
  await configure('auto');await page.click('#start');
  await page.waitForFunction(()=>window.floorLabStats.frames>=8,null,{timeout:120000});
  await mkdir('outputs/static',{recursive:true});
  await page.screenshot({path:'outputs/static/github-pages-mobile.png'});
  await page.click('#start');
  assert.deepEqual(errors,[]);assert.deepEqual(failures,[]);
  assert.ok(requests.has('vendor/ort-wasm-simd-threaded.asyncify.wasm'));
  assert.ok(requests.has('models/floor-fast.onnx'));
  assert.ok(requests.has('models/floor-quality.onnx'));
  assert.ok(requests.has('test-video.mp4'));
  results.requests=[...requests].sort();results.failures=failures;results.errors=errors;
  await writeFile('outputs/static/validation.json',JSON.stringify(results,null,2));
  console.log('Static Pages checks passed: nested path, no isolation headers, WASM single thread, WebGPU, both models, fallback, camera, no external requests.');
}finally{await browser.close();await new Promise(resolve=>server.close(resolve));}

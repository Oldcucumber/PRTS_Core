import {chromium} from 'playwright';
import assert from 'node:assert/strict';import {resolve} from 'node:path';import {mkdir,writeFile} from 'node:fs/promises';
// Requires locally generated WAV + camera fixture; no fixture ships in Pages.
process.env.BASE_PATH='/session-test/';const server=await(await import('../scripts/preview.mjs')).startPreview({port:0});const base='http://127.0.0.1:'+server.address().port+'/session-test/';
const seconds=Number(process.env.SESSION_SECONDS||75);
const browser=await chromium.launch({headless:true,channel:'chromium',args:['--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream','--use-file-for-fake-video-capture='+resolve(process.env.CAMERA_FIXTURE||'outputs/asr/camera.y4m'),'--use-file-for-fake-audio-capture='+resolve('outputs/asr/microphone.wav')]});
const context=await browser.newContext({permissions:['camera','microphone'],viewport:{width:390,height:844}}),page=await context.newPage();const errors=[],requests=[];page.on('pageerror',e=>errors.push(e.message));context.on('request',r=>{if(r.url().startsWith('http')&&!r.url().startsWith(base))requests.push(r.url());});
const observations=[];try{
 await page.goto(base);await page.waitForFunction(()=>window.prtsSession);assert.equal(await page.evaluate(()=>crossOriginIsolated),true);assert.equal(await page.locator('video').evaluate(v=>v.srcObject),null);
 await page.click('#start');await page.waitForFunction(()=>floorLabStats.frames>=3&&prtsSession.audio.ready,null,{timeout:120000});
 const voice=await page.evaluate(()=>prtsSession.speech.localVoice()?.name);assert.ok(voice,'local Chinese TTS required');
 const start=Date.now();let offline=false;
 while(Date.now()-start<seconds*1000){await page.waitForTimeout(5000);observations.push(await page.evaluate(()=>({time:performance.now(),frames:floorLabStats.frames,ms:floorLabStats.totalMs,audioReady:prtsSession.audio.ready,paused:prtsSession.core.paused,commands:prtsEvents.filter(e=>e.type==='command').map(e=>e.intent),playbacks:prtsEvents.filter(e=>e.type==='playback_start').length,errors:floorLabStats.errors,heap:performance.memory?.usedJSHeapSize})));if(!offline&&Date.now()-start>25000){await context.setOffline(true);offline=true;}}
 const report=await page.evaluate(()=>({frames:floorLabStats.frames,events:prtsEvents,errors:floorLabStats.errors,microphone:prtsSession.audio.stream?.active,isolated:crossOriginIsolated,overflow:document.documentElement.scrollWidth>innerWidth}));
 await mkdir('outputs/session',{recursive:true});await page.screenshot({path:'outputs/session/mobile.png'});await page.click('#settings');await page.screenshot({path:'outputs/session/settings.png'});await page.click('#close-settings');
 await writeFile('outputs/session/latest.json',JSON.stringify({report,observations,errors},null,2));
 const commands=report.events.filter(e=>e.type==='command').map(e=>e.intent);for(const command of ['pause','resume','mute','unmute'])assert.ok(commands.includes(command),'missing command '+command);
 assert.ok(report.events.some(e=>e.type==='playback_start'));assert.ok(report.events.some(e=>e.type==='playback_end'));assert.equal(report.overflow,false);assert.deepEqual(report.errors,[]);assert.deepEqual(errors,[]);assert.deepEqual(requests,[]);
 await page.evaluate(()=>{window.testTracks=[...prtsSession.audio.stream.getTracks(),...document.querySelector('video').srcObject.getTracks()];});await page.click('#start');assert.equal(await page.evaluate(()=>testTracks.every(t=>t.readyState==='ended')),true);assert.equal(await page.evaluate(()=>prtsSession.audio.worker===null&&!prtsSession.core.active),true);
 await writeFile('outputs/session/validation.json',JSON.stringify({input:'Recorded camera fixture via browser capture API; synthesized Chinese phrases via microphone capture API. Not an outdoor walking test.',durationSeconds:seconds,cameraFixture:process.env.CAMERA_FIXTURE||'still frame via capture API',offlineAfterSeconds:25,voice,report,observations,errors,requests},null,2));console.log('Continuous audio/video/TTS passed; duration '+seconds+'s; '+report.frames+' frames; commands '+commands.join(','));
}finally{await browser.close();await new Promise(r=>server.close(r));}

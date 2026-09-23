const bounded=promise=>Promise.race([promise,new Promise((_,reject)=>setTimeout(()=>reject(new Error('隔离初始化超时')),4000))]);
// Pages has no configurable HTTP headers. Reload once under our header-only SW.
if(!crossOriginIsolated&&navigator.serviceWorker){
 try{
 await bounded(navigator.serviceWorker.register(new URL('./isolation-sw.js',import.meta.url),{scope:'./'}));
 await bounded(navigator.serviceWorker.ready);
 if(!navigator.serviceWorker.controller)await bounded(new Promise(resolve=>navigator.serviceWorker.addEventListener('controllerchange',resolve,{once:true})));
 const key='prts-isolation-reload';
 if(!sessionStorage.getItem(key)){sessionStorage.setItem(key,'1');location.reload();await new Promise(()=>{});}
 }catch(error){console.warn('Local speech isolation unavailable',error);}
}
if(crossOriginIsolated)sessionStorage.removeItem('prts-isolation-reload');
await import('./app.mjs');await import('./session-ui.mjs');

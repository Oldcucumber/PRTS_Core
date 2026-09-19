// All browser assets are copied locally; no CDN or application server required.
import {mkdir,copyFile,writeFile,readFile,readdir,rm,realpath,stat} from 'node:fs/promises';
import {resolve,join,relative,sep} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';

const root=await realpath(fileURLToPath(new URL('..',import.meta.url)));
const output=resolve(root,'dist');
const entries=[
  ...['index.html','style.css','app.mjs','inference.worker.mjs','corridor.mjs'].map(name=>[`web/${name}`,name]),
  ...['fast','quality'].map(name=>[`web/models/floor-${name}.onnx`,`models/floor-${name}.onnx`]),
  // Non-isolated hosts select Asyncify; isolated development uses JSEP.
  ...['ort.webgpu.min.mjs','ort-wasm-simd-threaded.jsep.mjs','ort-wasm-simd-threaded.jsep.wasm','ort-wasm-simd-threaded.asyncify.mjs','ort-wasm-simd-threaded.asyncify.wasm'].map(name=>[`node_modules/onnxruntime-web/dist/${name}`,`vendor/${name}`]),
  ['VID20260919182406.mp4','test-video.mp4'],
  ['THIRD_PARTY_NOTICES.md','THIRD_PARTY_NOTICES.md'],
  ['licenses/onnxruntime-MIT.txt','licenses/onnxruntime-MIT.txt'],
  ['licenses/segformer-NVIDIA.txt','licenses/segformer-NVIDIA.txt'],
  ['licenses/segformer-model-card.md','licenses/segformer-model-card.md'],
];
// Validate prerequisites before replacing any previous successful build.
for(const [source] of entries){
  try{await stat(join(root,source));}
  catch{throw new Error(`Missing ${source}. Run npm ci; export missing models with python export_web.py.`);}
}
// Never follow an output-directory symlink or delete outside this project.
if(output!==join(root,'dist')||!output.startsWith(root+sep))throw new Error('Invalid build directory');
let existing;
try{existing=await realpath(output);}catch(error){if(error.code!=='ENOENT')throw error;}
if(existing&&existing!==output)throw new Error('Refusing to replace linked dist directory');
await rm(output,{recursive:true,force:true});
await mkdir(output,{recursive:true});
for(const [source,target] of entries){
  const destination=join(output,target);
  await mkdir(resolve(destination,'..'),{recursive:true});
  await copyFile(join(root,source),destination);
}
await writeFile(join(output,'.nojekyll'),'');
await writeFile(join(output,'DEPLOY.txt'),[
  'Deploy the CONTENTS of this directory to GitHub Pages or any HTTPS static host.',
  'All models, JavaScript, WASM and the test video are included. No CDN or backend is used.',
  'The entry point is index.html. Root and repository-subdirectory URLs are both supported.',
  'Use HTTPS for camera/WebGPU. file:// is not supported by module workers and camera APIs.',
  'GitHub Pages does not set COOP/COEP headers: WASM uses one CPU thread; WebGPU remains available.',
].join('\n')+'\n');
const manifest=[];
async function inventory(dir){
  for(const entry of await readdir(dir,{withFileTypes:true})){
    const path=join(dir,entry.name);
    if(entry.isDirectory())await inventory(path);
    else{
      const bytes=await readFile(path);
      manifest.push({path:relative(output,path).split(sep).join('/'),bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')});
    }
  }
}
await inventory(output);
await writeFile(join(output,'asset-manifest.json'),JSON.stringify({runtime:'onnxruntime-web@1.30.0',files:manifest},null,2)+'\n');
console.log(`Built ${manifest.length} local files, ${(manifest.reduce((sum,x)=>sum+x.bytes,0)/1024/1024).toFixed(1)} MiB -> dist/`);

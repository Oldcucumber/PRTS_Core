// Deliberately no COOP/COEP headers or application routes: mirrors Pages hosting.
import http from 'node:http';
import {createReadStream} from 'node:fs';
import {stat} from 'node:fs/promises';
import {resolve,extname,sep} from 'node:path';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('../dist/',import.meta.url));
const base='/' + (process.env.BASE_PATH||'').replace(/^\/+|\/+$/g,'') + '/';
const prefix=base==='//'?'/':base;
const types={'.html':'text/html; charset=utf-8','.css':'text/css','.mjs':'text/javascript','.js':'text/javascript','.json':'application/json','.mp4':'video/mp4','.wasm':'application/wasm','.onnx':'application/octet-stream','.txt':'text/plain; charset=utf-8','.md':'text/plain; charset=utf-8'};
export function startPreview({port=Number(process.env.PORT||8081),host=process.env.HOST||'127.0.0.1'}={}){
  return new Promise(resolveServer=>{
    const server=http.createServer(async(req,res)=>{
      try{
        if(!['GET','HEAD'].includes(req.method)){res.writeHead(405);res.end();return;}
        const url=new URL(req.url,'http://localhost');
        if(prefix!=='/'&&url.pathname===prefix.slice(0,-1)){res.writeHead(308,{Location:prefix+url.search});res.end();return;}
        const path=decodeURIComponent(url.pathname);
        if(!path.startsWith(prefix)){res.writeHead(404);res.end();return;}
        const file=resolve(root,path.slice(prefix.length)||'index.html');
        if(!file.startsWith(resolve(root)+sep)){res.writeHead(403);res.end();return;}
        const info=await stat(file);
        if(!info.isFile()){res.writeHead(404);res.end();return;}
        let start=0,end=info.size-1,code=200;
        if(req.headers.range){
          const match=/^bytes=(\d*)-(\d*)$/.exec(req.headers.range);
          if(!match||(!match[1]&&!match[2]))throw new Error('Invalid range');
          if(match[1]){start=Number(match[1]);end=match[2]?Math.min(Number(match[2]),end):end;}
          else start=Math.max(0,info.size-Number(match[2]));
          if(start>end||start>=info.size){res.writeHead(416,{'Content-Range':`bytes */${info.size}`});res.end();return;}
          code=206;res.setHeader('Content-Range',`bytes ${start}-${end}/${info.size}`);
        }
        res.writeHead(code,{'Content-Type':types[extname(file)]||'application/octet-stream','Content-Length':Math.max(0,end-start+1),'Accept-Ranges':'bytes','Cache-Control':'no-cache'});
        if(req.method==='HEAD'||info.size===0)res.end();
        else createReadStream(file,{start,end}).on('error',()=>res.destroy()).pipe(res);
      }catch{res.writeHead(404);res.end('Not found');}
    });
    server.listen(port,host,()=>{console.log(`Static preview: http://${host}:${server.address().port}${prefix}`);resolveServer(server);});
  });
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))await startPreview();

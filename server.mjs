import http from 'node:http';
import https from 'node:https';
import { createReadStream, readFileSync, statSync } from 'node:fs';
import { resolve, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('.', import.meta.url));
const web = resolve(root, 'web');
const vendor = resolve(root, 'node_modules/onnxruntime-web/dist');
const mime = { '.html':'text/html; charset=utf-8', '.js':'text/javascript', '.mjs':'text/javascript', '.css':'text/css', '.json':'application/json', '.wasm':'application/wasm', '.onnx':'application/octet-stream', '.mp4':'video/mp4', '.svg':'image/svg+xml' };
function handler(req, res) {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  try {
    if (!['GET','HEAD'].includes(req.method)) { res.writeHead(405); res.end(); return; }
    const path = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
    let file;
    {
      const base = path.startsWith('/vendor/') ? vendor : web;
      const relative = path.startsWith('/vendor/') ? path.slice(8) : path === '/' ? 'index.html' : path.slice(1);
      file = resolve(base, relative);
      if (!file.startsWith(base + sep)) { res.writeHead(403); res.end(); return; }
    }
    const size = statSync(file).size;
    res.setHeader('Content-Type', mime[extname(file)] || 'application/octet-stream');
    res.setHeader('Accept-Ranges', 'bytes');
    res.setHeader('Cache-Control', /\.(onnx|wasm)$/.test(file) ? 'public, max-age=86400' : 'no-cache');
    let start = 0, end = size - 1, status = 200;
    if (req.headers.range) {
      const match = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range);
      if (!match || (!match[1] && !match[2])) { res.writeHead(416); res.end(); return; }
      if (match[1]) { start = Number(match[1]); end = match[2] ? Math.min(Number(match[2]), end) : end; }
      else start = Math.max(0, size-Number(match[2]));
      if (start > end || start >= size) { res.writeHead(416, {'Content-Range':`bytes */${size}`}); res.end(); return; }
      status = 206;
      res.setHeader('Content-Range', `bytes ${start}-${end}/${size}`);
    }
    res.setHeader('Content-Length', end-start+1);
    res.writeHead(status);
    if (req.method === 'HEAD') res.end();
    else createReadStream(file, {start,end}).on('error', () => res.destroy()).pipe(res);
  } catch { res.writeHead(404); res.end('Not found'); }
}
const secure = process.env.TLS_KEY && process.env.TLS_CERT;
const server = secure ? https.createServer({ key:readFileSync(process.env.TLS_KEY), cert:readFileSync(process.env.TLS_CERT) },handler) : http.createServer(handler);
server.listen(Number(process.env.PORT || 8080), process.env.HOST || '0.0.0.0', () => {
  console.log(`Floor lab: ${secure?'https':'http'}://localhost:${server.address().port}`);
  console.log('Phone cameras require HTTPS with a trusted certificate.');
});

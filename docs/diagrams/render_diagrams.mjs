// VeriFace AI diagram export: renders every docs/diagrams/*.mmd to docs/assets/diagrams/<name>.svg and <name>.png.
// Headless Chrome/Edge over the DevTools protocol (Node >= 22, no npm packages), same approach as
// docs/project_guide/build/render.mjs. The browser downloads only the Mermaid 11.4.1 library from jsDelivr;
// the diagram sources are rendered locally and never sent anywhere.
//   node docs/diagrams/render_diagrams.mjs
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.resolve(HERE, "..", "assets", "diagrams");
const CHROME = ["C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe", "/usr/bin/google-chrome", "/usr/bin/chromium"].find(fs.existsSync);
const PORT = 9335;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

if (!CHROME) throw new Error("Chrome or Edge not found");
fs.mkdirSync(OUT, { recursive: true });
const sources = fs.readdirSync(HERE).filter((f) => f.endsWith(".mmd"));

const profile = fs.mkdtempSync(path.join(os.tmpdir(), "veriface-diagram-"));
const proc = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  "--allow-file-access-from-files", "--no-first-run", "--disable-extensions", "about:blank"], { stdio: "ignore" });
try {
  let ver;
  for (let i = 0; i < 50 && !ver; i++) { try { ver = await (await fetch(`http://127.0.0.1:${PORT}/json/version`)).json(); } catch { await sleep(200); } }
  if (!ver) throw new Error("Chrome did not start");
  const ws = new WebSocket(ver.webSocketDebuggerUrl);
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
  let id = 0; const pending = new Map();
  ws.onmessage = (ev) => { const m = JSON.parse(ev.data); if (m.id && pending.has(m.id)) { const { r, j } = pending.get(m.id); pending.delete(m.id); m.error ? j(new Error(JSON.stringify(m.error))) : r(m.result); } };
  const send = (method, params = {}, sessionId) => new Promise((r, j) => { const i = ++id; pending.set(i, { r, j }); ws.send(JSON.stringify({ id: i, method, params, sessionId })); });
  const { targetId } = await send("Target.createTarget", { url: "about:blank" });
  const { sessionId } = await send("Target.attachToTarget", { targetId, flatten: true });
  const p = (m, a) => send(m, a, sessionId);
  const ev = async (expr) => { const r = await p("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true }); if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails)); return r.result.value; };
  await p("Page.enable"); await p("Runtime.enable");

  for (const file of sources) {
    const name = path.basename(file, ".mmd");
    const src = fs.readFileSync(path.join(HERE, file), "utf8");
    const html = `<!doctype html><meta charset="utf-8"><body style="margin:0;background:#fff">
<div id="out" style="display:inline-block;padding:16px;background:#fff"></div>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js"></script>
<script>
mermaid.initialize({ startOnLoad:false, theme:"base", securityLevel:"strict",
  themeVariables:{ fontFamily:"Segoe UI, Arial, sans-serif", fontSize:"15px", lineColor:"#555" },
  flowchart:{ htmlLabels:false, useMaxWidth:false, curve:"basis", nodeSpacing:28, rankSpacing:38, padding:10 },
  sequence:{ useMaxWidth:false, mirrorActors:false } });
mermaid.render("d", ${JSON.stringify(src)}).then(({svg}) => { document.getElementById("out").innerHTML = svg; window.__done = true; })
  .catch(e => { window.__err = String(e); });
</script>`;
    const tmp = path.join(os.tmpdir(), `veriface_${name}.html`); fs.writeFileSync(tmp, html);
    await p("Emulation.setDeviceMetricsOverride", { width: 2400, height: 2400, deviceScaleFactor: 2, mobile: false });
    await p("Page.navigate", { url: pathToFileURL(tmp).href });
    const t = Date.now();
    while (!(await ev("!!(window.__done || window.__err)").catch(() => false))) { if (Date.now() - t > 60000) throw new Error("timeout " + name); await sleep(200); }
    const err = await ev("window.__err || ''"); if (err) throw new Error(`${file}: ${err}`);
    const svg = await ev(`(() => { const s = document.querySelector('#out svg'); s.setAttribute('xmlns','http://www.w3.org/2000/svg');
      return new XMLSerializer().serializeToString(s); })()`);
    fs.writeFileSync(path.join(OUT, `${name}.svg`), svg);
    const box = await ev(`(() => { const r = document.getElementById('out').getBoundingClientRect(); return {x:r.x,y:r.y,w:r.width,h:r.height}; })()`);
    const shot = await p("Page.captureScreenshot", { format: "png", captureBeyondViewport: true, clip: { x: box.x, y: box.y, width: box.w, height: box.h, scale: 1 } });
    fs.writeFileSync(path.join(OUT, `${name}.png`), Buffer.from(shot.data, "base64"));
    fs.rmSync(tmp);
    console.log(`${name}: ${Math.round(box.w)} x ${Math.round(box.h)} css px -> ${path.relative(process.cwd(), OUT)}`);
  }
  ws.close();
} finally {
  proc.kill();
  try { fs.rmSync(profile, { recursive: true, force: true }); } catch {}
}

// VeriFace AI guide build tool: headless Chrome over the DevTools protocol (Node >= 22, no npm packages).
//   node render.mjs diagram   -> ../architecture.svg, ../architecture.png   (Mermaid 11.4.1 from jsDelivr)
//   node render.mjs pdf       -> ../VeriFace_AI_A_to_Z_Guide.pdf           (Markdown via marked 15.0.7)
//   node render.mjs preview N1,N2,...  -> preview/page_N.png  (renders PDF pages with pdf.js 4.10.38, for checking)
// Needs network access to cdn.jsdelivr.net. Chrome profile goes to a temp folder; the user's browser is untouched.
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const GUIDE = path.resolve(HERE, "..");
const CHROME = ["C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"].find(fs.existsSync);
const PORT = 9333;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function withChrome(fn) {
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), "veriface-chrome-"));
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
    const page = {
      send: (m, p) => send(m, p, sessionId),
      async goto(url) { await page.send("Page.enable"); await page.send("Runtime.enable"); await page.send("Page.navigate", { url }); },
      async eval(expr) { const r = await page.send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true }); if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails)); return r.result.value; },
      async waitFor(expr, ms = 60000) { const t = Date.now(); while (Date.now() - t < ms) { const v = await page.eval(expr).catch(() => null); if (v) return v; await sleep(250); } throw new Error("timeout waiting for " + expr); },
    };
    await fn(page);
    ws.close();
  } finally { proc.kill(); await sleep(500); try { fs.rmSync(profile, { recursive: true, force: true }); } catch {} }
}

async function diagram() {
  const src = fs.readFileSync(path.join(GUIDE, "architecture.mmd"), "utf8");
  const html = `<!doctype html><meta charset="utf-8"><body style="margin:0;background:#fff">
<div id="out"></div>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js"></script>
<script>
mermaid.initialize({ startOnLoad:false, theme:"base", securityLevel:"loose",
  themeVariables:{ fontFamily:"Segoe UI, Arial, sans-serif", fontSize:"15px", lineColor:"#555" },
  flowchart:{ htmlLabels:true, curve:"basis", nodeSpacing:28, rankSpacing:38, padding:10 } });
mermaid.render("arch", ${JSON.stringify(src)}).then(({svg}) => { document.getElementById("out").innerHTML = svg; window.__done = true; })
  .catch(e => { window.__err = String(e); });
</script>`;
  const tmp = path.join(HERE, "_diagram.html"); fs.writeFileSync(tmp, html);
  await withChrome(async (page) => {
    await page.goto(pathToFileURL(tmp).href);
    await page.waitFor("window.__done || window.__err");
    const err = await page.eval("window.__err || ''"); if (err) throw new Error(err);
    const svg = await page.eval(`(() => { const s = document.querySelector('#out svg');
      s.setAttribute('xmlns','http://www.w3.org/2000/svg'); return new XMLSerializer().serializeToString(s); })()`);
    fs.writeFileSync(path.join(GUIDE, "architecture.svg"), svg);
    const box = await page.eval(`(() => { const r = document.querySelector('#out svg').getBoundingClientRect(); return {x:r.x,y:r.y,w:r.width,h:r.height}; })()`);
    await page.send("Emulation.setDeviceMetricsOverride", { width: Math.ceil(box.w) + 20, height: Math.ceil(box.h) + 20, deviceScaleFactor: 2, mobile: false });
    await sleep(300);
    const shot = await page.send("Page.captureScreenshot", { format: "png", clip: { x: box.x, y: box.y, width: box.w, height: box.h, scale: 1 } });
    fs.writeFileSync(path.join(GUIDE, "architecture.png"), Buffer.from(shot.data, "base64"));
    console.log("diagram", Math.round(box.w), "x", Math.round(box.h), "css px");
  });
  fs.rmSync(tmp);
}

async function pdf() {
  const md = fs.readFileSync(path.join(GUIDE, "VeriFace_AI_A_to_Z_Guide.md"), "utf8");
  const css = fs.readFileSync(path.join(HERE, "guide.css"), "utf8");
  const tp = path.join(HERE, "toc_pages.json");
  const tocPages = fs.existsSync(tp) ? JSON.parse(fs.readFileSync(tp, "utf8")) : {};
  const html = `<!doctype html><html><head><meta charset="utf-8"><title>VeriFace AI — A-to-Z Project Guide</title>
<base href="${pathToFileURL(GUIDE + path.sep).href}"><style>${css}</style></head><body><main id="doc"></main>
<script src="https://cdn.jsdelivr.net/npm/marked@15.0.7/marked.min.js"></script>
<script>
const md = ${JSON.stringify(md)};
const doc = document.getElementById("doc");
doc.innerHTML = marked.parse(md, { gfm: true });
// GitHub-style heading ids so the Markdown table-of-contents links also work in the PDF
const seen = {};
for (const h of doc.querySelectorAll("h1,h2,h3,h4")) {
  let s = h.textContent.trim().toLowerCase().replace(/[^\\p{L}\\p{N}\\s-]/gu, "").replace(/\\s/g, "-");
  if (seen[s] !== undefined) s += "-" + (++seen[s]); else seen[s] = 0;
  h.id = s;
}
// page-break control: every H2 starts a new page except the first after the cover
doc.querySelectorAll("h2").forEach((h, i) => { if (i > 0) h.classList.add("newpage"); });
// optional second pass: page numbers measured from the first PDF render (build/toc_pages.json)
const tocPages = ${JSON.stringify(tocPages)};
const tocList = document.querySelector('#contents + ol');
if (tocList) { tocList.classList.add("toc"); for (const li of tocList.children) { const a = li.querySelector("a");
  const n = a && tocPages[a.getAttribute("href").slice(1)]; if (n) li.insertAdjacentHTML("beforeend", '<span class="toc-page">' + n + '</span>'); } }
const imgs = [...document.images];
Promise.all(imgs.map(i => i.complete ? 1 : new Promise(r => { i.onload = i.onerror = r; }))).then(() => { window.__ready = true; });
</script></body></html>`;
  const tmp = path.join(HERE, "_guide.html"); fs.writeFileSync(tmp, html);
  await withChrome(async (page) => {
    await page.goto(pathToFileURL(tmp).href);
    await page.waitFor("window.__ready");
    const broken = await page.eval("[...document.images].filter(i => !i.naturalWidth).map(i => i.src)");
    if (broken.length) throw new Error("broken images: " + broken.join(", "));
    const footer = `<div style="font-family:Segoe UI,Arial,sans-serif;font-size:8px;color:#666;width:100%;padding:0 14mm;display:flex;justify-content:space-between">
      <span>VeriFace AI — A-to-Z Project Guide</span><span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span></div>`;
    const r = await page.send("Page.printToPDF", { printBackground: true, preferCSSPageSize: true, displayHeaderFooter: true,
      headerTemplate: "<div></div>", footerTemplate: footer, generateDocumentOutline: true, generateTaggedPDF: true });
    const out = path.join(GUIDE, "VeriFace_AI_A_to_Z_Guide.pdf");
    fs.writeFileSync(out, Buffer.from(r.data, "base64"));
    console.log("pdf bytes", fs.statSync(out).size);
  });
  fs.rmSync(tmp);
}

async function preview(pages) {
  const pdfPath = path.join(GUIDE, "VeriFace_AI_A_to_Z_Guide.pdf");
  const b64 = fs.readFileSync(pdfPath).toString("base64");
  const outDir = path.join(HERE, "preview"); fs.mkdirSync(outDir, { recursive: true });
  const html = `<!doctype html><meta charset="utf-8"><body><script type="module">
import * as pdfjs from "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.min.mjs";
pdfjs.GlobalWorkerOptions.workerSrc = "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.worker.min.mjs";
const data = Uint8Array.from(atob(${JSON.stringify(b64)}), c => c.charCodeAt(0));
const pdf = await pdfjs.getDocument({ data }).promise;
window.__n = pdf.numPages;
window.__render = async (n) => { const p = await pdf.getPage(n); const v = p.getViewport({ scale: 1.4 });
  const c = document.createElement("canvas"); c.width = v.width; c.height = v.height;
  await p.render({ canvasContext: c.getContext("2d"), viewport: v }).promise;
  const t = (await p.getTextContent()).items.map(i => i.str).join(" ");
  return { png: c.toDataURL("image/png").split(",")[1], chars: t.length }; };
window.__text = async () => { let s = ""; for (let i = 1; i <= pdf.numPages; i++) s += (await (await pdf.getPage(i)).getTextContent()).items.map(x => x.str).join(" ") + "\\n\\f"; return s; };
window.__ok = true;
</script>`;
  const tmp = path.join(HERE, "_preview.html"); fs.writeFileSync(tmp, html);
  await withChrome(async (page) => {
    await page.goto(pathToFileURL(tmp).href);
    await page.waitFor("window.__ok");
    const n = await page.eval("window.__n");
    console.log("pages", n);
    fs.writeFileSync(path.join(outDir, "pdf_text.txt"), await page.eval("window.__text()"));
    for (const p of pages.map(x => x === "last" ? n : Number(x))) {
      const { png, chars } = await page.eval(`window.__render(${p})`);
      fs.writeFileSync(path.join(outDir, `page_${p}.png`), Buffer.from(png, "base64"));
      console.log("page", p, "text chars", chars);
    }
  });
  fs.rmSync(tmp);
}

const [mode, arg] = process.argv.slice(2);
if (mode === "diagram") await diagram();
else if (mode === "pdf") await pdf();
else if (mode === "preview") await preview((arg || "1,2,last").split(","));
else { console.error("usage: node render.mjs diagram|pdf|preview [pages]"); process.exit(2); }

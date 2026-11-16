// Convert output/figure*.svg to vector PDFs (same size, text kept as text) for the LaTeX build.
// Needs Playwright: run `npm install` in this folder (see package.json), then
//     node figures/svg_to_pdf.js
// Uses Playwright's Chromium if installed, otherwise an installed Edge or Chrome.
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');
const { chromium } = require('playwright');
const OUT = path.join(__dirname, 'output');

async function launch() {
  for (const opts of [{}, { channel: 'msedge' }, { channel: 'chrome' }]) {
    try { return await chromium.launch(opts); } catch (e) { /* next */ }
  }
  throw new Error('no Chromium-based browser found: run `npx playwright install chromium`');
}

(async () => {
  const b = await launch();
  const p = await b.newPage();
  const files = fs.readdirSync(OUT).filter(f => /^figure.*\.svg$/.test(f)).sort();
  for (const f of files) {
    const svg = fs.readFileSync(path.join(OUT, f), 'utf8');
    const m = svg.match(/viewBox="0 0 ([\d.]+) ([\d.]+)"/);
    const [w, h] = [parseFloat(m[1]), parseFloat(m[2])];
    await p.setViewportSize({ width: Math.ceil(w), height: Math.ceil(h) });
    await p.goto(pathToFileURL(path.join(OUT, f)).href);
    const out = path.join(OUT, f.replace(/\.svg$/, '.pdf'));
    await p.pdf({ path: out, width: `${w}px`, height: `${h}px`, printBackground: true,
                  margin: { top: 0, right: 0, bottom: 0, left: 0 }, pageRanges: '1' });
    console.log(f, '->', path.basename(out), `${w}x${h}`);
  }
  await b.close();
})();

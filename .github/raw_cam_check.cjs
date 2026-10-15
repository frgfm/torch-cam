const fs = require('fs');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE);
(async () => {
  fs.mkdirSync('raw-cam-check', { recursive: true });
  const browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
  page.setDefaultTimeout(120000);
  const report = {};
  try {
    await page.goto('https://frgfm-torch-cam.hf.space/', { waitUntil: 'domcontentloaded' });
    await page.getByRole('heading', { name: 'TorchCAM Explorer', exact: true }).waitFor();
    await page.getByRole('combobox', { name: 'Classification model', exact: true }).waitFor();
    await page.waitForTimeout(500);
    await page.locator('input[type=file]').setInputFiles(process.env.DEMO_CHECK_IMAGE);
    await page.getByRole('button', { name: 'Compute CAM', exact: true }).click();
    await page.getByRole('button', { name: 'Download raw tensor', exact: true }).waitFor();
    await page.getByRole('button', { name: 'Download overlay', exact: true }).waitFor();
    const raw = page.locator('.st-key-raw-cam img');
    await raw.waitFor();
    await raw.evaluate(img => img.decode());
    report.raw = await raw.evaluate(img => ({ width: img.naturalWidth, height: img.naturalHeight, rendering: getComputedStyle(img).imageRendering, source: img.currentSrc }));
    const response = await page.request.get(report.raw.source);
    report.raw.contentType = response.headers()['content-type'];
    if (report.raw.rendering !== 'pixelated') throw new Error('Raw map is interpolated by the browser');
    if (!report.raw.contentType.startsWith('image/png')) throw new Error('Raw map is not lossless PNG');
    if (report.raw.width !== 7 || report.raw.height !== 7) throw new Error('Default ResNet CAM no longer has its native grid');
    await page.mouse.move(0, 0);
    await page.screenshot({ path: 'raw-cam-check/image-classifier-grid.png', fullPage: true });
    console.log(JSON.stringify(report));
  } catch (error) {
    report.error = error.message;
    await page.screenshot({ path: 'raw-cam-check/failure.png', fullPage: true }).catch(() => {});
    throw error;
  } finally {
    fs.writeFileSync('raw-cam-check/report.json', JSON.stringify(report, null, 2));
    await browser.close();
  }
})().catch(error => { console.error(error.message); process.exitCode = 1; });

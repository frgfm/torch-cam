const fs = require('fs');
const path = require('path');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE);

(async () => {
  const directory = 'live-demo-check';
  fs.mkdirSync(directory, { recursive: true });
  const browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
  page.setDefaultTimeout(180000);
  const report = {};
  try {
    await page.goto('https://frgfm-torch-cam.hf.space/', { waitUntil: 'domcontentloaded' });
    await page.getByRole('heading', { name: 'TorchCAM Explorer' }).waitFor();
    await page.getByRole('tab', { name: 'VLMs', exact: true }).click();
    await page.locator('input[type=file]').setInputFiles(process.env.DEMO_CHECK_IMAGE);
    let started = Date.now();
    await page.getByRole('button', { name: 'Generate caption', exact: true }).click();
    await page.getByText(/LayerCAM · raw next-token probability/).waitFor();
    await page.getByRole('heading', { name: 'Overlay', exact: true }).waitFor();
    report.vlm = { seconds: (Date.now() - started) / 1000, method: await page.getByText(/LayerCAM · raw next-token probability/).innerText() };
    await page.screenshot({ path: path.join(directory, 'vlm-collie.png'), fullPage: true });
    await page.getByRole('tab', { name: 'Decision models', exact: true }).click();
    const ticket = page.getByRole('textbox', { name: 'Support ticket', exact: true });
    await ticket.fill('I was charged twice for the same order. Please refund the duplicate payment.');
    await ticket.blur();
    started = Date.now();
    await page.getByRole('button', { name: 'Explain decision', exact: true }).click();
    const laya = page.getByText(/Laya · Token ablation on class scores/);
    await laya.waitFor();
    report.laya = { seconds: (Date.now() - started) / 1000, method: await laya.innerText() };
    if (await page.locator('[data-testid="stHtml"]').count() !== 3) throw new Error('Laya did not render all decision maps');
    await page.screenshot({ path: path.join(directory, 'decision-laya.png'), fullPage: true });
    await page.getByRole('combobox', { name: 'Decision model', exact: true }).click();
    await page.getByRole('option', { name: 'MiniLM · zero-shot classifier', exact: true }).click();
    started = Date.now();
    await page.getByRole('button', { name: 'Explain decision', exact: true }).click();
    const minilm = page.getByText(/MiniLM.*Token ablation on class scores/);
    await minilm.waitFor();
    report.minilm = { seconds: (Date.now() - started) / 1000, method: await minilm.innerText() };
    if (await page.locator('[data-testid="stHtml"]').count() !== 3) throw new Error('MiniLM did not render all decision maps');
    await page.screenshot({ path: path.join(directory, 'decision-minilm.png'), fullPage: true });
    console.log(JSON.stringify(report));
  } catch (error) {
    report.error = error.message;
    await page.screenshot({ path: path.join(directory, 'failure.png'), fullPage: true }).catch(() => {});
    throw error;
  } finally {
    fs.writeFileSync(path.join(directory, 'report.json'), JSON.stringify(report, null, 2));
    await browser.close();
  }
})().catch(error => { console.error(error.message); process.exitCode = 1; });

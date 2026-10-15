const fs = require('fs');
const path = require('path');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE);

(async () => {
  const directory = 'nav-demo-check';
  fs.mkdirSync(directory, { recursive: true });
  const browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });
  page.setDefaultTimeout(180000);
  const report = {};
  try {
    await page.goto('https://frgfm-torch-cam.hf.space/', { waitUntil: 'domcontentloaded' });
    await page.getByRole('heading', { name: 'TorchCAM Explorer' }).waitFor();
    const checkTabs = async () => {
      for (const name of ['Image classifiers', 'VLMs', 'Decision models']) {
        if (!await page.getByRole('tab', { name, exact: true }).isVisible()) throw new Error('Missing task tab: ' + name);
      }
    };
    await checkTabs();
    await page.getByRole('combobox', { name: 'Classification model', exact: true }).waitFor();
    await page.waitForTimeout(500);
    await page.locator('input[type=file]').setInputFiles(process.env.DEMO_CHECK_IMAGE);
    await page.getByRole('button', { name: 'Compute CAM', exact: true }).click();
    await page.getByRole('button', { name: 'Download overlay', exact: true }).waitFor();
    await page.locator('.st-key-raw-cam img').evaluate(img => img.decode());
    await page.screenshot({ path: path.join(directory, 'image-classifier.png'), fullPage: true });
    await page.getByRole('tab', { name: 'VLMs', exact: true }).click();
    await checkTabs();
    await page.getByRole('combobox', { name: 'VLM model', exact: true }).waitFor();
    await page.getByRole('combobox', { name: 'VLM model', exact: true }).click();
    await page.getByRole('option', { name: 'BLIP captioner', exact: true }).click();
    report.vlmSelector = await page.locator('.st-key-vlm-model').innerText();
    await page.waitForTimeout(500);
    await page.locator('input[type=file]').setInputFiles(process.env.DEMO_CHECK_IMAGE);
    let started = Date.now();
    await page.getByRole('button', { name: 'Generate caption', exact: true }).click();
    await page.getByRole('combobox', { name: 'Caption token', exact: true }).click();
    await page.getByRole('option', { name: /· 'dog'$/ }).click();
    await page.getByText(/LayerCAM · raw next-token probability/).waitFor();
    await page.getByRole('heading', { name: 'Overlay', exact: true }).waitFor();
    await page.getByRole('button', { name: 'Download token map', exact: true }).waitFor();
    await page.locator('[data-testid="stImage"] img').last().waitFor();
    await page.waitForFunction(() => [...document.querySelectorAll('[data-testid="stImage"] img')].every(img => img.complete && img.naturalWidth > 0));
    report.vlm = { seconds: (Date.now() - started) / 1000, method: await page.getByText(/LayerCAM · raw next-token probability/).innerText() };
    await page.screenshot({ path: path.join(directory, 'vlm-collie.png'), fullPage: true });
    await page.getByRole('tab', { name: 'Decision models', exact: true }).click();
    await checkTabs();
    await page.getByRole('combobox', { name: 'Decision model', exact: true }).waitFor();
    const ticket = page.getByRole('textbox', { name: 'Support ticket', exact: true });
    await ticket.fill('I was charged twice for the same order. Please refund the duplicate payment.');
    await ticket.blur();
    started = Date.now();
    await page.getByRole('button', { name: 'Explain decision', exact: true }).click();
    const laya = page.getByText(/^Laya · Token ablation on class scores · [\d.]+ s$/);
    await laya.waitFor();
    report.laya = { seconds: (Date.now() - started) / 1000, method: await laya.innerText() };
    await page.waitForFunction(() => [...document.querySelectorAll('[data-testid="stHtml"]')].filter(node => node.textContent.includes('I was charged twice')).length === 3);
    await page.getByRole('button', { name: 'Download signed contributions', exact: true }).waitFor();
    await page.waitForTimeout(250);
    await page.screenshot({ path: path.join(directory, 'decision-laya.png'), fullPage: true });
    await page.getByRole('combobox', { name: 'Decision model', exact: true }).click();
    await page.getByRole('option', { name: 'MiniLM · zero-shot classifier', exact: true }).click();
    await laya.waitFor({ state: 'hidden' });
    started = Date.now();
    await page.getByRole('button', { name: 'Explain decision', exact: true }).click();
    const minilm = page.getByText(/^MiniLM · zero-shot classifier · Token ablation on class scores · [\d.]+ s$/);
    await minilm.waitFor();
    report.minilm = { seconds: (Date.now() - started) / 1000, method: await minilm.innerText() };
    await page.waitForFunction(() => [...document.querySelectorAll('[data-testid="stHtml"]')].filter(node => node.textContent.includes('I was charged twice')).length === 3);
    await page.getByRole('button', { name: 'Download signed contributions', exact: true }).waitFor();
    await page.waitForTimeout(250);
    await page.screenshot({ path: path.join(directory, 'decision-minilm.png'), fullPage: true });
    await page.getByRole('tab', { name: 'VLMs', exact: true }).click();
    await page.getByRole('combobox', { name: 'VLM model', exact: true }).waitFor();
    await checkTabs();
    if (!(await page.locator('.st-key-vlm-model').innerText()).includes('BLIP captioner')) throw new Error('VLM selection was lost');
    console.log(JSON.stringify(report));
  } catch (error) {
    report.error = error.message;
    report.testIds = await page.locator('[data-testid]').evaluateAll(nodes => nodes.reduce((counts, node) => { const id = node.dataset.testid; counts[id] = (counts[id] || 0) + 1; return counts; }, {}));
    await page.screenshot({ path: path.join(directory, 'failure.png'), fullPage: true }).catch(() => {});
    throw error;
  } finally {
    fs.writeFileSync(path.join(directory, 'report.json'), JSON.stringify(report, null, 2));
    await browser.close();
  }
})().catch(error => { console.error(error.message); process.exitCode = 1; });

const fs = require('fs');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE);
(async () => {
 const browser = await chromium.launch({headless:true,args:['--no-sandbox']});
 const page = await browser.newPage({viewport:{width:1440,height:1100}});
 page.setDefaultTimeout(60000);
 fs.mkdirSync('family-ui',{recursive:true});
 try {
  await page.goto('http://127.0.0.1:8080/');
  await page.getByRole('heading',{name:'TorchCAM Explorer'}).waitFor();
  await page.getByRole('tab',{name:'VLMs',exact:true}).click();
  const vlm=page.getByRole('combobox',{name:'VLM model',exact:true});
  await vlm.waitFor();await vlm.click();
  await page.getByRole('option',{name:'Qwen3-VL · 2B',exact:true}).click();
  await page.waitForTimeout(500);
  await page.locator('input[type=file]').setInputFiles('fixtures/border-collie.jpg');
  await page.getByRole('button',{name:'Generate caption',exact:true}).waitFor();
  await page.waitForTimeout(500);
  await vlm.click();
  for(const name of ['BLIP captioner','Qwen3-VL · 2B','Gemma 4 · E2B']) await page.getByRole('option',{name,exact:true}).waitFor();
  await page.screenshot({path:'family-ui/vlm-models.png',fullPage:true});
  await page.getByRole('option',{name:'Gemma 4 · E2B',exact:true}).click();
  await page.waitForTimeout(500);
  if(await vlm.inputValue()!=='Gemma 4 · E2B')throw Error('Gemma selector did not persist');
  await page.getByRole('tab',{name:'Decision models',exact:true}).click();
  const decision=page.getByRole('combobox',{name:'Decision model',exact:true});
  await decision.click();
  for(const name of ['Laya','MiniLM · zero-shot classifier','DeBERTa · zero-shot classifier','Decider · 0.8B']) await page.getByRole('option',{name,exact:true}).waitFor();
  await page.screenshot({path:'family-ui/decision-models.png',fullPage:true});
  await page.getByRole('option',{name:'DeBERTa · zero-shot classifier',exact:true}).click();
  await page.getByRole('tab',{name:'VLMs',exact:true}).click();
  await vlm.waitFor();
  if(await vlm.inputValue()!=='Gemma 4 · E2B')throw Error('Navigation reset model choice');
  for(const name of ['Image classifiers','VLMs','Decision models'])await page.getByRole('tab',{name,exact:true}).waitFor();
 } catch(error) {await page.screenshot({path:'family-ui/failure.png',fullPage:true});throw error;}
 finally {await browser.close();}
})();

const fs = require('fs');
const {chromium} = require(process.env.PLAYWRIGHT_PACKAGE);
(async()=>{
 const directory='live-family-check';fs.mkdirSync(directory,{recursive:true});
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1440,height:1100}});page.setDefaultTimeout(600000);
 const report={source:process.env.PREVIEW_SOURCE_SHA,vlms:{},decisions:{}};
 const save=()=>fs.writeFileSync(`${directory}/report.json`,JSON.stringify(report,null,2));
 const screenshot=async(name)=>{await page.waitForFunction(()=>[...document.querySelectorAll('[data-testid="stImage"] img')].every(i=>i.complete&&i.naturalWidth>0));await page.screenshot({path:`${directory}/${name}.png`,fullPage:true});save();};
 const choose=async(label,value)=>{const box=page.getByRole('combobox',{name:label,exact:true});await box.click();await page.getByRole('option',{name:value,exact:true}).click();await page.waitForTimeout(500);};
 try{
  await page.goto('https://frgfm-torch-cam.hf.space/',{waitUntil:'domcontentloaded'});
  await page.getByRole('heading',{name:'TorchCAM Explorer'}).waitFor();
  const tabs=async()=>{for(const name of ['Image classifiers','VLMs','Decision models'])await page.getByRole('tab',{name,exact:true}).waitFor();};await tabs();
  await page.getByRole('tab',{name:'VLMs',exact:true}).click();
  const vlm=page.getByRole('combobox',{name:'VLM model',exact:true});await vlm.click();
  for(const name of ['BLIP captioner','Qwen3-VL · 2B','Gemma 4 · E2B'])await page.getByRole('option',{name,exact:true}).waitFor();
  await screenshot('vlm-models');await page.getByRole('option',{name:'Qwen3-VL · 2B',exact:true}).click();
  await page.locator('input[type=file]').setInputFiles('fixtures/border-collie.jpg');
  await page.waitForFunction(()=>[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Generate caption'&&!b.disabled));
  for(const [name,word,method] of [['Qwen3-VL · 2B',/· '\s*(Border|dog)'$/i,'GradCAM'],['Gemma 4 · E2B',/· '\s*(dog|border)'$/i,'LayerCAM']]){
   if(name.startsWith('Gemma'))await choose('VLM model',name);
   const button=page.getByRole('button',{name:'Generate caption',exact:true});
   if(await button.isDisabled()){
    const message=await page.getByText('Gemma needs a GPU or a BF16-accelerated CPU. Choose BLIP or Qwen on this server.',{exact:true}).innerText();
    report.vlms[name]={available:false,reason:message};await screenshot('gemma-hardware');continue;
   }
   const start=Date.now();await button.click();
   const tokenBox=page.getByRole('combobox',{name:'Caption token',exact:true});
   await Promise.race([tokenBox.waitFor(),page.getByText('The caption could not be generated. Check model availability and available memory, then retry.',{exact:true}).waitFor().then(()=>{throw Error(name+': caption generation failed')})]);
   report.current={model:name,phase:'caption ready',text:await page.locator('[data-testid=stMain]').innerText()};console.log(JSON.stringify(report.current));save();await screenshot(name.startsWith('Qwen')?'qwen-caption':'gemma-caption');
   await tokenBox.click();
   console.log('Available caption tokens:',JSON.stringify(await page.getByRole('option').allTextContents()));
   await page.getByRole('option',{name:word}).first().click({timeout:5000});
   await page.getByText(new RegExp(`${method} · raw next-token probability`)).waitFor({timeout:180000});
   await page.getByRole('button',{name:'Download token map',exact:true}).waitFor({timeout:10000});
   report.vlms[name]={available:true,seconds:(Date.now()-start)/1000,method:await page.getByText(new RegExp(`${method} · raw next-token probability`)).innerText()};
   await screenshot(name.startsWith('Qwen')?'qwen-collie':'gemma-collie');console.log(name,JSON.stringify(report.vlms[name]));
  }
  await page.getByRole('tab',{name:'Decision models',exact:true}).click();await tabs();
  const decision=page.getByRole('combobox',{name:'Decision model',exact:true});await decision.click();
  for(const name of ['Laya','MiniLM · zero-shot classifier','DeBERTa · zero-shot classifier','Decider · 0.8B'])await page.getByRole('option',{name,exact:true}).waitFor();
  await screenshot('decision-models');await page.getByRole('option',{name:'DeBERTa · zero-shot classifier',exact:true}).click();
  const ticket=page.getByRole('textbox',{name:'Support ticket',exact:true});await ticket.fill('I was charged twice for the same order. Please refund the duplicate payment.');await ticket.blur();
  for(const name of ['DeBERTa · zero-shot classifier','Decider · 0.8B']){
   if(name.startsWith('Decider'))await choose('Decision model',name);
   const start=Date.now();await page.getByRole('button',{name:'Explain decision',exact:true}).click();
   await Promise.race([page.getByText(`${name} · Token ablation on class scores`,{exact:false}).waitFor(),page.getByText('The decision could not be explained. Check model availability and available memory, then retry.',{exact:true}).waitFor().then(()=>{throw Error(name+': decision explanation failed')})]);
   await page.getByRole('button',{name:'Download signed contributions',exact:true}).waitFor();
   await page.waitForFunction(()=>[...document.querySelectorAll('[data-testid="stHtml"]')].filter(n=>n.textContent.includes('I was charged twice')).length===3);
   report.decisions[name]={seconds:(Date.now()-start)/1000};await screenshot(name.startsWith('Decider')?'decision-decider':'decision-deberta');console.log(name,JSON.stringify(report.decisions[name]));
  }
  await page.getByRole('tab',{name:'Image classifiers',exact:true}).click();await tabs();
  await page.getByRole('combobox',{name:'Classification model',exact:true}).waitFor();report.navigation=true;save();
 }catch(error){report.error=error.message;report.visible=await page.locator('[data-testid=stMain]').innerText().catch(()=>null);report.alerts=await page.locator('[data-testid="stAlert"]').allTextContents().catch(()=>[]);await page.screenshot({path:`${directory}/failure.png`,fullPage:true}).catch(()=>{});throw error;}
 finally{save();await browser.close();}
})().catch(e=>{console.error(e.message);process.exitCode=1});

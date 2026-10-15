const fs=require('fs');const {chromium}=require(process.env.PLAYWRIGHT_PACKAGE);
(async()=>{
 const which=process.env.MODEL_CASE,dir=`live-runtime-${which}`;fs.mkdirSync(dir,{recursive:true});
 const report={case:which,source:'85482d69acf8c7be78a0a6b7658aa732fb227e38',models:{}};
 const save=()=>fs.writeFileSync(`${dir}/report.json`,JSON.stringify(report,null,2));
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});const page=await browser.newPage({viewport:{width:1440,height:1100}});page.setDefaultTimeout(300000);
 const shot=async(name)=>{await page.screenshot({path:`${dir}/${name}.png`,fullPage:true});save()};
 const choose=async(label,value)=>{await page.getByRole('combobox',{name:label,exact:true}).click();await page.getByRole('option',{name:value,exact:true}).click();await page.waitForTimeout(500)};
 try{
  await page.goto('https://frgfm-torch-cam.hf.space/',{waitUntil:'domcontentloaded'});await page.getByRole('heading',{name:'TorchCAM Explorer'}).waitFor();
  for(const name of ['Image classifiers','VLMs','Decision models'])await page.getByRole('tab',{name,exact:true}).waitFor();
  if(which==='decisions'){
   await page.getByRole('tab',{name:'Decision models',exact:true}).click();
   await page.getByRole('combobox',{name:'Decision model',exact:true}).click();
   for(const name of ['Laya','MiniLM · zero-shot classifier','DeBERTa · zero-shot classifier','Decider · 0.8B'])await page.getByRole('option',{name,exact:true}).waitFor();
   await page.getByRole('option',{name:'DeBERTa · zero-shot classifier',exact:true}).click();
   const ticket=page.getByRole('textbox',{name:'Support ticket',exact:true});await ticket.fill('I was charged twice for the same order. Please refund the duplicate payment.');await ticket.blur();
   for(const name of ['DeBERTa · zero-shot classifier','Decider · 0.8B']){
    if(name.startsWith('Decider'))await choose('Decision model',name);
    report.phase=name;save();const started=Date.now();await page.getByRole('button',{name:'Explain decision',exact:true}).click();
    await Promise.race([page.getByText(`${name} · Token ablation on class scores`,{exact:false}).waitFor(),page.getByText('The decision could not be explained. Check model availability and available memory, then retry.',{exact:true}).waitFor().then(()=>{throw Error(name+' explanation failed')})]);
    await page.getByRole('button',{name:'Download signed contributions',exact:true}).waitFor();
    await page.waitForFunction(()=>[...document.querySelectorAll('[data-testid="stHtml"]')].filter(n=>n.textContent.includes('I was charged twice')).length===3);
    report.models[name]={seconds:(Date.now()-started)/1000};await shot(name.startsWith('Decider')?'decider':'deberta');console.log(name,JSON.stringify(report.models[name]));
   }
  }else{
   const gemma=which==='gemma',name=gemma?'Gemma 4 · E2B':'Qwen3-VL · 2B',method=gemma?'LayerCAM':'GradCAM';
   await page.getByRole('tab',{name:'VLMs',exact:true}).click();await choose('VLM model',name);
   const hint=page.getByText('Gemma needs a GPU or a BF16-accelerated CPU. Choose BLIP or Qwen on this server.',{exact:true});
   await page.locator('input[type=file]').setInputFiles('fixtures/border-collie.jpg');
   const enabled=()=>page.waitForFunction(()=>[...document.querySelectorAll('button')].some(b=>b.textContent.trim()==='Generate caption'&&!b.disabled));
   const available=gemma?await Promise.race([hint.waitFor().then(()=>false),enabled().then(()=>true)]):await enabled().then(()=>true);
   if(!available){report.models[name]={available:false,reason:await hint.innerText()};await shot('gemma-hardware');return;}
   report.phase='generate';save();const started=Date.now();await page.getByRole('button',{name:'Generate caption',exact:true}).click();
   const box=page.getByRole('combobox',{name:'Caption token',exact:true});
   await Promise.race([box.waitFor(),page.getByText('The caption could not be generated. Check model availability and available memory, then retry.',{exact:true}).waitFor().then(()=>{throw Error(name+' generation failed')})]);
   report.phase='caption ready';report.caption=await page.locator('[data-testid=stMain]').innerText();save();await shot('caption');console.log(report.caption);
   await box.click();report.tokens=await page.getByRole('option').allTextContents();save();console.log(JSON.stringify(report.tokens));
   await page.getByRole('option',{name:/· '\s*(dog|border)'$/i}).first().click({timeout:5000});
   report.phase='map';save();await page.getByText(new RegExp(`${method} · raw next-token probability`)).waitFor({timeout:180000});
   await page.getByRole('button',{name:'Download token map',exact:true}).waitFor({timeout:10000});
   await page.waitForFunction(()=>[...document.querySelectorAll('[data-testid="stImage"] img')].every(i=>i.complete&&i.naturalWidth>0),null,{timeout:10000});
   report.models[name]={available:true,seconds:(Date.now()-started)/1000,method:await page.getByText(new RegExp(`${method} · raw next-token probability`)).innerText()};await shot('map');console.log(JSON.stringify(report.models[name]));
  }
  report.phase='complete';save();
 }catch(error){report.error=error.message;report.visible=await page.locator('[data-testid=stMain]').innerText().catch(()=>null);await shot('failure').catch(()=>{});throw error;}
 finally{save();await browser.close()}
})().catch(e=>{console.error(e.message);process.exitCode=1});

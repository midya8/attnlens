'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),{spawn,execFileSync}=require('node:child_process'),{waitForDebugger}=require('./chrome_support.js');
const delay=ms=>new Promise(r=>setTimeout(r,ms));
function compare(a,b,p='root'){assert.equal(typeof a,typeof b,p);if(typeof a==='number')assert.ok(Number.isFinite(b)&&Math.abs(a-b)<=1e-11*Math.max(1,Math.abs(a),Math.abs(b)),p);else if(a===null||typeof a!=='object')assert.equal(a,b,p);else{assert.deepEqual(Object.keys(a).sort(),Object.keys(b).sort(),p);for(const k of Object.keys(a))compare(a[k],b[k],p+'.'+k);}}
test('actual Chrome full intervention reports, every displayed tensor and weight, exports and invalidation',{timeout:180000},async()=>{
 const server=require('node:http').createServer((req,res)=>{try{const file=path.join(__dirname,'../docs',req.url==='/'?'index.html':req.url);const types={'.html':'text/html','.js':'application/javascript','.css':'text/css','.json':'application/json'};res.setHeader('Content-Type',types[path.extname(file)]||'text/plain');res.end(fs.readFileSync(file));}catch(e){res.statusCode=404;res.end('not found');}});await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const profile=fs.mkdtempSync(path.join(os.tmpdir(),'attnlens-audit-')),browser=spawn(process.env.CHROME_BIN||'google-chrome',['--headless','--no-sandbox','--disable-gpu','--disable-extensions','--disable-dev-shm-usage','--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'],{stdio:['ignore','ignore','pipe']});let ws,startupError,diagnostics='';browser.on('error',e=>startupError=e);browser.stderr.on('data',d=>diagnostics=(diagnostics+d).slice(-4000));
 try{for(let i=0;i<1200&&!startupError&&browser.exitCode===null&&!fs.existsSync(path.join(profile,'DevToolsActivePort'));i++)await delay(25);assert.ok(fs.existsSync(path.join(profile,'DevToolsActivePort')),startupError?.message||diagnostics||'Chrome unavailable');const port=fs.readFileSync(path.join(profile,'DevToolsActivePort'),'utf8').split('\n')[0],tab=await waitForDebugger(`http://127.0.0.1:${port}/json/list`,{alive:()=>browser.exitCode===null&&!startupError});ws=new WebSocket(tab.webSocketDebuggerUrl);await new Promise((res,rej)=>{ws.onopen=res;ws.onerror=rej;});let seq=0;const pending=new Map();ws.onmessage=e=>{const m=JSON.parse(e.data);if(pending.has(m.id)){const {res,rej}=pending.get(m.id);pending.delete(m.id);m.error?rej(Error(JSON.stringify(m.error))):res(m.result);}};const call=(method,params={})=>new Promise((res,rej)=>{const id=++seq;pending.set(id,{res,rej});ws.send(JSON.stringify({id,method,params}));});const js=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
 await call('Page.enable');await call('Page.navigate',{url:process.env.ATTN_URL||`http://127.0.0.1:${server.address().port}/`});
 for(let i=0;i<500;i++){if(await js('Boolean(window.workbenchReady)'))break;await delay(20);}assert.ok(await js('window.workbenchReady'));
 const native=JSON.parse(execFileSync(process.env.PYTHON||'python',['tests/native_reports.py'],{cwd:path.resolve(__dirname,'..'),env:{...process.env,PYTHONPATH:'.'},maxBuffer:80e6}));
 const data=require('../docs/data.json'),fmt=x=>typeof x==='number'?(Number.isInteger(x)?String(x):Number(x).toFixed(6)):String(x);
 const tables=async id=>js(`Array.from(document.querySelectorAll('#'+${JSON.stringify(id)}+' tbody'),body=>Array.from(body.rows,tr=>Array.from(tr.cells,c=>c.textContent)))`);
 const expected=rows=>rows.map(row=>row.map(fmt));
 const matrix=rows=>expected(rows.map((row,i)=>[i,...row]));
 const set=async(id,value,event='change')=>js(`document.getElementById(${JSON.stringify(id)}).value=${JSON.stringify(String(value))};document.getElementById(${JSON.stringify(id)}).dispatchEvent(new Event(${JSON.stringify(event)}))`);
 let headViews=0;
 for(let mi=0;mi<data.models.length;mi++){
  await set('model',mi);
  for(let ci=0;ci<data.cases.length;ci++){
   await set('fixture',ci);const item=native[mi*data.cases.length+ci];
   for(let mask=0;mask<16;mask++){
    await js(`document.querySelector('[data-mask="${mask}"]').click()`);const r=item.reports[mask];compare(r,await js('currentAudit.intervention'));
    const got=await tables('logits'),b=item.reports[0];assert.deepEqual(got,[expected(b.probabilities.map((p,i)=>[i,b.logits.at(-1)[i],r.logits.at(-1)[i],p,r.probabilities[i]]))]);
   }
   await js(`document.querySelector('[data-mask="2"]').click()`);compare(item.audit,await js('currentAudit'));
   for(let li=0;li<2;li++){await set('layer',li);for(let h=0;h<2;h++){await set('head',h);for(let row=0;row<data.cases[ci].tokens.length;row++){
    await set('row',row);headViews++;const L=item.audit.intervention.layers[li],B=item.audit.baseline.layers[li],p=data.models[mi].model,dh=p.dim/p.heads;
    assert.deepEqual(await tables('attention'),[expected(data.cases[ci].tokens.map((t,i)=>[i,t,B.attention[h][row][i],L.attention[h][row][i]]))]);
    assert.deepEqual(await tables('qkv'),[matrix([L.q[h][row]]),matrix(L.k[h]),matrix(L.v[h]),expected(L.concat[row].slice(h*dh,(h+1)*dh).map((v,i)=>[i,v,L.effective[row][h*dh+i]]))]);
    assert.deepEqual(await tables('residual'),[matrix(item.audit.intervention.embedded),...['input','concat','effective','projected','residual'].map(k=>matrix(L[k])),matrix(item.audit.intervention.logits)]);
   }}}
   assert.deepEqual(await tables('weights'),Object.values(data.models[mi].model.groups).map(matrix));
   assert.deepEqual(await tables('counterfactuals'),[expected(item.audit.counterfactuals.map(c=>[c.kind,c.position,c.target,c.prediction,c.target_probability,c.loss]))]);
  }
  await js('document.getElementById("evaluation").closest("details").open=true');await delay(20);
  assert.deepEqual(await tables('evaluation'),[expected(data.models[mi].evaluation.map(r=>[r.mask,(r.accuracy*100).toFixed(3)+'%',r.mean_loss,r.mean_loss-data.models[mi].evaluation[0].mean_loss]))]);
 }
 for(const bad of ['','10,2,11,11','true,10,2,11','10,2,1,','10,2,1,11x','10,2,1,12','10,2.5,1,11']){await set('tokens',bad,'input');assert.equal(await js('document.getElementById("results").hidden'),true);assert.equal(await js('document.getElementById("export").disabled'),true);await js('document.getElementById("apply").click()');assert.equal(await js('document.getElementById("results").hidden'),true);}
 await set('model',0);await set('fixture',0);await js(`document.querySelector('[data-mask="2"]').click()`);await js('HTMLAnchorElement.prototype.click=function(){window.downloaded={name:this.download,url:this.href}};URL.revokeObjectURL=function(){}');
 for(const id of ['export','checkpoint']){await js(`document.getElementById('${id}').click()`);const output=await js('fetch(downloaded.url).then(r=>r.json())');if(id==='export')compare({schema:1,checkpoint:data.models[0].model,...native[0].audit,intervention:native[0].reports[2]},output);else compare(data.models[0].model,output);}
 await set('tokens',data.cases[1].tokens.join(','),'input');await js('document.getElementById("tokens").focus()');await call('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});await call('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',windowsVirtualKeyCode:13});assert.equal(await js('document.getElementById("results").hidden'),false);
 for(const width of [320,390,1280]){await call('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:width<700});assert.ok(await js('document.documentElement.scrollWidth<=innerWidth'),'page overflow '+width);assert.ok(await js(`Array.from(document.querySelectorAll('button,input:not([type=checkbox]),select,summary')).filter(e=>e.getClientRects().length).every(e=>e.getBoundingClientRect().height>=44)`),'touch '+width);}
 console.log(JSON.stringify({complete_reports:352,full_audits:22,head_row_views:headViews,exports:2,widths:[320,390,1280]}));
 if(process.env.ATTN_SCREENSHOT){const width=Number(process.env.ATTN_WIDTH||1280);await call('Emulation.setDeviceMetricsOverride',{width,height:1200,deviceScaleFactor:1,mobile:width<700});await set('model',0);await set('fixture',0);await js(`document.querySelector('[data-mask="2"]').click();document.activeElement.blur();window.scrollTo(0,0)`);if(process.env.ATTN_EXPAND)await js('document.querySelectorAll("details").forEach(e=>e.open=true)');const m=await call('Page.getLayoutMetrics'),image=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip:{x:0,y:0,width,height:m.cssContentSize.height,scale:1}});fs.writeFileSync(process.env.ATTN_SCREENSHOT,Buffer.from(image.data,'base64'));}
 }finally{server.close();if(ws)ws.close();if(browser.pid&&browser.exitCode===null){browser.kill('SIGTERM');await new Promise(r=>browser.once('exit',r));}await fs.promises.rm(profile,{recursive:true,force:true,maxRetries:10,retryDelay:200});}
});

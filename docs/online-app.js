/* DOM presentation only. The independent engine computes all intermediates. */
'use strict';
const $=id=>document.getElementById(id),fmt=x=>x===null?'empty':typeof x==='number'?(Number.isInteger(x)?String(x):Math.abs(x)<.00001&&x!==0?x.toExponential(5):x.toFixed(6)):String(x);
let payload,currentSpec,currentAudit;
function node(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;}
function table(headers,rows){const t=node('table'),head=node('thead'),tr=node('tr');for(const h of headers)tr.append(node('th',h));head.append(tr);t.append(head);const body=node('tbody');for(const row of rows){const tr=node('tr');for(const v of row)tr.append(node('td',fmt(v)));body.append(tr);}t.append(body);return t;}
function tensor(title,rows){const box=node('div');box.append(node('p',title,'hint'));box.append(table(['row',...rows[0].map((_,i)=>'d'+i)],rows.map((r,i)=>[i,...r])));return box;}
function select(id,items,value=0){$(id).replaceChildren(...items.map(([v,label])=>{const o=node('option',label);o.value=v;return o;}));$(id).value=String(value);}
function metrics(id,items){$(id).replaceChildren(...items.map(([label,value,hint])=>{const m=node('div',undefined,'metric');m.append(node('small',label),node('strong',value),node('span',hint));return m;}));}
function invalidate(){currentSpec=currentAudit=null;window.onlineSpec=window.onlineAudit=null;$('results').hidden=true;for(const id of ['download-input','download-audit'])$(id).disabled=true;$('status').className='';$('status').textContent='Inputs changed. Run inference to replace the previous result.';}
function run(){
 try{
  const raw=$('tokens').value.trim();if(!/^\d+(?:\s*,\s*\d+)*$/.test(raw))throw Error('Use comma-separated integer token IDs, no trailing comma.');
  const target=$('target').value;if(!/^\d+$/.test(target))throw Error('Target must be an integer token ID.');
  const spec={schema:OnlineEngine.SCHEMA,model:payload.models[Number($('checkpoint').value)].model,tokens:raw.split(',').map(x=>Number(x.trim())),target:Number(target),block_size:Number($('block').value),order:$('order').value};
  const result=OnlineEngine.audit(spec);currentSpec=spec;currentAudit=result;window.onlineSpec=spec;window.onlineAudit=result;
  const oldLayer=Number($('layer').value)||0,oldHead=Number($('head').value)||0,oldRow=$('row').value===''?spec.tokens.length-1:Math.min(Number($('row').value),spec.tokens.length-1);
  select('layer',Array.from({length:spec.model.layers},(_,i)=>[i,'Layer '+i]),Math.min(oldLayer,spec.model.layers-1));select('head',Array.from({length:spec.model.heads},(_,i)=>[i,'Head '+i]),Math.min(oldHead,spec.model.heads-1));select('row',spec.tokens.map((t,i)=>[i,i+' / token '+t]),oldRow);
  const a=result.online_summary,b=result.dense_summary;
  metrics('metrics',[
   ['Online prediction',a.prediction,'Dense prediction '+b.prediction],['Online loss',fmt(a.loss),'Dense '+fmt(b.loss)+' nats'],['Target probability',fmt(a.target_probability),'Dense '+fmt(b.target_probability)],
   ['Max absolute logit error',fmt(result.max_absolute_logit_error),'All positions × vocabulary'],['Max scaled logit error',fmt(result.max_scaled_logit_error),'No exact-equality claim'],['Legacy scaled logit error',fmt(result.legacy_max_scaled_logit_error),'Separate finite-sentinel control']]);
  $('logits').replaceChildren(table(['token','online logit','dense logit','online P','dense P'],a.probabilities.map((p,i)=>[i,result.online.logits.at(-1)[i],result.dense.logits.at(-1)[i],p,b.probabilities[i]])));
  $('all-logits').replaceChildren(table(['position','token','online','dense','difference'],result.online.logits.flatMap((r,i)=>r.map((x,c)=>[i,c,x,result.dense.logits[i][c],x-result.dense.logits[i][c]]))));
  const s=result.online.stats;
  metrics('counts',[
   ['Peak live score entries',s.peak_score_buffer,'Observed list length; cap '+spec.block_size],['Score evaluations',s.score_evaluations,'Causal pairs × heads × layers'],['Blocks consumed',s.blocks,'Includes partial prefixes'],['Full score-matrix slots',s.score_matrix_slots,'Inference core only'],['Full probability-matrix slots',s.probability_matrix_slots,'Inference core only'],['Numerator entries',s.numerator_slots,'One active head; plus scalar m and l']]);
  $('results').hidden=false;for(const id of ['download-input','download-audit'])$(id).disabled=false;
  $('status').className='';$('status').textContent='Complete source weights → online Q/K/V → head results → output projection → residuals → all logits. Dense audit recomputed separately.';renderRow();window.onlineReady=true;
 }catch(error){invalidate();$('status').className='error';$('status').textContent=error.message;}
}
function renderRow(){if(!currentAudit)return;
 const li=Number($('layer').value),hi=Number($('head').value),r=Number($('row').value),L=currentAudit.online.layers[li],D=currentAudit.dense.layers[li],h=L.heads[hi],d=D.heads[hi],row=h.rows[r],last=row.steps.at(-1),tokens=currentSpec.tokens;
 $('caption').textContent='Layer '+li+', head '+hi+', query '+r+': '+(r+1)+' admitted keys, '+row.steps.length+' blocks in '+currentSpec.order+' order. Final m = '+fmt(last.maximum)+', l = '+fmt(last.normalizer)+'.';
 $('weights').replaceChildren(...tokens.map((t,j)=>{const n=node('div',undefined,'key'+(j>r?' future':''));const space=node('div',undefined,'bar-space'),bar=node('div',undefined,'bar');bar.style.height=(d.attention[r][j]*100)+'px';space.append(bar);n.append(node('small',j>r?'future':d.attention[r][j]===0?'0':d.attention[r][j]<.001?d.attention[r][j].toExponential(0):d.attention[r][j].toFixed(3)),space,node('small',j+' / '+t));n.title='Key '+j+', token '+t+(j>r?': excluded':': dense P = '+d.attention[r][j]);return n;}));
 const headers=['block / keys','previous m','previous l','rescale α','new m','block mass','new l',...last.numerator.map((_,i)=>'new u'+i),...last.output.map((_,i)=>'u/l '+i)];
 $('transitions').replaceChildren(table(headers,row.steps.map((s,i)=>[i+' / '+s.keys.join(','),s.previous_max,s.previous_normalizer,s.rescale,s.maximum,s.block_sum,s.normalizer,...s.numerator,...s.output])));
 $('output').replaceChildren(table(['dimension','online head','dense head','difference'],row.output.map((x,i)=>[i,x,d.output[r][i],x-d.output[r][i]])));
 const keyScores=new Map();for(const s of row.steps)for(let i=0;i<s.keys.length;i++)keyScores.set(s.keys[i],s.scores[i]);
 $('qkv').replaceChildren(tensor('Actual query Q', [h.q[r]]),table(['key','token','status','score','dense P',...h.k[0].map((_,i)=>'K'+i),...h.v[0].map((_,i)=>'V'+i)],tokens.map((t,j)=>[j,t,j<=r?'admitted':'excluded',j<=r?keyScores.get(j):'not computed',d.attention[r][j],...h.k[j],...h.v[j]])));
 $('residual').replaceChildren(...['input','concat','projected','residual'].flatMap(key=>[tensor('Online '+key,L[key]),tensor('Dense '+key,D[key])]));
}
function fixture(){const c=payload.cases[Number($('fixture').value)];$('tokens').value=c.tokens.join(',');$('target').value=c.target;$('row').replaceChildren();run();}
function download(kind){if(!currentAudit)return;const blob=new Blob([JSON.stringify(kind==='input'?currentSpec:currentAudit,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),a=node('a');a.href=url;a.download='attnlens-online-'+kind+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);}
(async()=>{try{
 const response=await fetch('data.json');if(!response.ok)throw Error('Checkpoint fetch failed: '+response.status);payload=await response.json();
 select('checkpoint',payload.models.map((m,i)=>[i,m.label]));select('fixture',payload.cases.map((c,i)=>[i,c.id+' / '+c.tokens.length+' tokens']));select('block',Array.from({length:24},(_,i)=>[i+1,String(i+1)]),3);
 $('fixture').addEventListener('change',fixture);for(const id of ['checkpoint','block','order'])$(id).addEventListener('change',run);
 for(const id of ['tokens','target']){$(id).addEventListener('input',invalidate);$(id).addEventListener('keydown',e=>{if(e.key==='Enter')run();});}
 for(const id of ['layer','head','row'])$(id).addEventListener('change',renderRow);$('apply').addEventListener('click',run);$('download-input').addEventListener('click',()=>download('input'));$('download-audit').addEventListener('click',()=>download('audit'));fixture();
 }catch(error){$('status').className='error';$('status').textContent=error.message;}})();

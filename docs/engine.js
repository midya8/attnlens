/* Independent fixed-weight inference. No Python-generated intermediate tensors. */
(function(root){
'use strict';
const integer=(x,lo,hi,name)=>{if(!Number.isInteger(x)||x<lo||x>hi)throw Error(`${name} must be integer ${lo}..${hi}`);return x;};
const exact=(x,keys)=>{if(!x||Array.isArray(x)||typeof x!=='object'||Object.keys(x).sort().join('|')!==keys.slice().sort().join('|'))throw Error('Unexpected model fields');};
function model(data){
 exact(data,['schema','vocab','dim','heads','layers','groups']);if(data.schema!==1)throw Error('Unsupported model schema');
 const v=integer(data.vocab,2,32,'vocab'),d=integer(data.dim,2,16,'dim'),h=integer(data.heads,1,4,'heads'),l=integer(data.layers,1,3,'layers');
 if(d%h)throw Error('dim must be divisible by heads');
 const shapes={embed:[v,d],wout:[d,v],bout:[1,v]};
 for(let i=0;i<l;i++){shapes[`l${i}_wo`]=[d,d];for(let k=0;k<h;k++)for(const q of ['wq','wk','wv'])shapes[`l${i}_${q}${k}`]=[d,d/h];}
 exact(data.groups,Object.keys(shapes));
 for(const [k,[r,c]]of Object.entries(shapes)){const a=data.groups[k];if(!Array.isArray(a)||a.length!==r||a.some(row=>!Array.isArray(row)||row.length!==c||row.some(x=>typeof x!=='number'||!Number.isFinite(x)||Math.abs(x)>1e4)))throw Error('Invalid parameter shape/value');}
 return data;
}
const mm=(a,b)=>a.map(row=>b[0].map((_,j)=>row.reduce((s,x,i)=>s+x*b[i][j],0)));
const sm=row=>{const m=Math.max(...row),e=row.map(x=>Math.exp(x-m)),s=e.reduce((x,y)=>x+y,0);return e.map(x=>x/s);};
const ce=(row,t)=>{const m=Math.max(...row);return(m-row[t])+Math.log(row.reduce((s,x)=>s+Math.exp(x-m),0));};
const argmax=row=>row.reduce((best,x,i)=>x>row[best]?i:best,0);
function run(tokens,target,p,muted=[]){
 model(p);if(!Array.isArray(tokens)||tokens.length<1||tokens.length>24)throw Error('tokens must contain 1..24 integers');tokens.forEach(x=>integer(x,0,p.vocab-1,'token'));integer(target,0,p.vocab-1,'target');
 if(!Array.isArray(muted))throw Error('muted must be an array');const blocked=new Set();
 for(const pair of muted){if(!Array.isArray(pair)||pair.length!==2)throw Error('Invalid muted pair');const li=integer(pair[0],0,p.layers-1,'layer'),h=integer(pair[1],0,p.heads-1,'head'),key=`${li}:${h}`;if(blocked.has(key))throw Error('Duplicate head');blocked.add(key);}
 const g=p.groups,d=p.dim,dh=d/p.heads,L=tokens.length;
 let x=tokens.map((t,r)=>g.embed[t].map((e,i)=>e+(i%2?Math.cos(r/10000**(2*Math.floor(i/2)/d)):Math.sin(r/10000**(2*Math.floor(i/2)/d)))));
 const embedded=x.map(row=>row.slice()),layers=[];
 for(let li=0;li<p.layers;li++){
  const qs=[],ks=[],vs=[],attention=[],heads=[];
  for(let h=0;h<p.heads;h++){
   const q=mm(x,g[`l${li}_wq${h}`]),k=mm(x,g[`l${li}_wk${h}`]),v=mm(x,g[`l${li}_wv${h}`]);
   const a=q.map((row,i)=>sm(k.map((kr,j)=>j>i?-1e30:row.reduce((s,u,z)=>s+u*kr[z],0)/Math.sqrt(dh))));
   qs.push(q);ks.push(k);vs.push(v);attention.push(a);heads.push(mm(a,v));
  }
  const concat=x.map((_,r)=>heads.flatMap(h=>h[r]));
  const effective=concat.map(row=>row.map((v,i)=>blocked.has(`${li}:${Math.floor(i/dh)}`)?0:v));
  const projected=mm(effective,g[`l${li}_wo`]);
  const residual=x.map((row,r)=>row.map((v,i)=>v+projected[r][i]));
  layers.push({input:x,q:qs,k:ks,v:vs,attention,concat,effective,projected,residual});x=residual;
 }
 const logits=mm(x,g.wout).map(row=>row.map((v,i)=>v+g.bout[0][i])),row=logits[L-1],probabilities=sm(row),prediction=argmax(row);
 return{tokens:tokens.slice(),target,muted:[...blocked].map(k=>k.split(':').map(Number)).sort((a,b)=>a[0]-b[0]||a[1]-b[1]),embedded,layers,logits,probabilities,prediction,correct:prediction===target,loss:ce(row,target),target_probability:probabilities[target],margin:row[target]-Math.max(...row.filter((_,i)=>i!==target))};
}
function summary(r){return Object.fromEntries(['prediction','correct','loss','target_probability','margin'].map(k=>[k,r[k]]));}
function sweep(tokens,target,p){
 if(p.layers*p.heads>8)throw Error('Subset sweep limited to 8 heads');const base=run(tokens,target,p),rows=[];
 for(let mask=0;mask<2**(p.layers*p.heads);mask++){const muted=[];for(let i=0;i<p.layers*p.heads;i++)if(mask&(1<<i))muted.push([Math.floor(i/p.heads),i%p.heads]);const r=summary(run(tokens,target,p,muted));rows.push({...r,mask,muted,loss_delta:r.loss-base.loss,probability_delta:r.target_probability-base.target_probability});}return rows;
}
function counterfactuals(tokens){
 if(!Array.isArray(tokens)||tokens.length<4||tokens.filter(t=>t===10).length!==1||tokens.at(-1)!==11||tokens.some(t=>!Number.isInteger(t)||t<0||t>=12)||tokens.slice(0,-1).includes(11))throw Error('Expected one marker and final query');
 const marker=tokens.indexOf(10);if(marker>=tokens.length-2)throw Error('Marker must precede a key');const key=marker+1,rows=[];
 for(let t=0;t<10;t++){const changed=tokens.slice();changed[key]=t;rows.push({kind:'key',position:key,tokens:changed,target:t});}
 for(let i=0;i<tokens.length-1;i++){if(i===marker||i===key)continue;const changed=tokens.slice();changed[i]=(changed[i]+1)%10;rows.push({kind:'filler',position:i,tokens:changed,target:tokens[key]});}return rows;
}
function inspect(tokens,target,p,muted=[]){return{baseline:run(tokens,target,p),intervention:run(tokens,target,p,muted),subsets:sweep(tokens,target,p),counterfactuals:counterfactuals(tokens).map(v=>({...v,...summary(run(v.tokens,v.target,p))}))};}
const api={model,run,inspect,sweep,counterfactuals,summary,ce};if(typeof module!=='undefined')module.exports=api;else root.AttnEngine=api;
})(typeof window!=='undefined'?window:globalThis);

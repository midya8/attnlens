/* Independent binary64 implementation of standard online causal attention.
   Core uses one score block. Saved traces/dense audit are intentionally quadratic. */
(function(root){
'use strict';
const Legacy=typeof module!=='undefined'?require('./engine.js'):root.AttnEngine;
const SCHEMA='attnlens-online/v1';
function integer(x,lo,hi,name){if(!Number.isInteger(x)||x<lo||x>hi)throw Error(name+' must be integer '+lo+'..'+hi);return x;}
function exact(x,keys){if(!x||typeof x!=='object'||Array.isArray(x)||Object.keys(x).sort().join('|')!==keys.slice().sort().join('|'))throw Error('Unexpected fields');}
function matrix(x,name,rows,cols){
 if(!Array.isArray(x)||x.length<1||x.length>24||(rows!==undefined&&rows!==x.length))throw Error(name+' invalid rows');
 const w=Array.isArray(x[0])?x[0].length:0;if(w<1||w>16||(cols!==undefined&&cols!==w))throw Error(name+' invalid columns');
 for(const r of x)if(!Array.isArray(r)||r.length!==w||r.some(v=>typeof v!=='number'||!Number.isFinite(v)||Math.abs(v)>1e100))throw Error(name+' invalid values');return w;
}
function options(block,order){integer(block,1,24,'block_size');if(!['forward','reverse'].includes(order))throw Error('Invalid block order');}
function attention(q,k,v,block_size=4,order='forward',trace=false){
 const d=matrix(q,'q'),dv=matrix(v,'v',q.length);matrix(k,'k',q.length,d);options(block_size,order);if(typeof trace!=='boolean')throw Error('trace must be boolean');
 const output=[],rows=[],stats={score_evaluations:0,blocks:0,peak_score_buffer:0,score_matrix_slots:0,probability_matrix_slots:0,numerator_slots:dv},scale=1/Math.sqrt(d);
 for(let i=0;i<q.length;i++){
  let maximum=null,normalizer=0;const numerator=Array(dv).fill(0),steps=[],count=Math.ceil((i+1)/block_size);
  for(let turn=0;turn<count;turn++){
   const start=(order==='forward'?turn:count-1-turn)*block_size,end=Math.min(i+1,start+block_size);
   let scores=[];for(let j=start;j<end;j++){let score=0;for(let c=0;c<d;c++)score+=q[i][c]*k[j][c];scores.push(score*scale);}
   stats.peak_score_buffer=Math.max(stats.peak_score_buffer,scores.length);stats.score_evaluations+=scores.length;stats.blocks++;
   let next=maximum===null?Math.max(...scores):Math.max(maximum,...scores),alpha=maximum===null?0:Math.exp(maximum-next);
   const previous=trace?{previous_max:maximum,previous_normalizer:normalizer,previous_numerator:numerator.slice()}:null;
   const weights=trace?[]:null,weighted=trace?Array(dv).fill(0):null;
   normalizer*=alpha;for(let c=0;c<dv;c++)numerator[c]*=alpha;let sum=0;
   for(let offset=0;offset<scores.length;offset++){
    const weight=Math.exp(scores[offset]-next);sum+=weight;
    for(let c=0;c<dv;c++){const term=weight*v[start+offset][c];numerator[c]+=term;if(trace)weighted[c]+=term;}
    if(trace)weights.push(weight);
   }
   normalizer+=sum;maximum=next;
   if(trace)steps.push({...previous,keys:Array.from({length:end-start},(_,j)=>start+j),scores:scores.slice(),weights,block_sum:sum,block_weighted:weighted,rescale:alpha,maximum,normalizer,numerator:numerator.slice(),output:numerator.map(x=>x/normalizer)});
   scores=null;
  }
  const result=numerator.map(x=>x/normalizer);output.push(result);if(trace)rows.push({query:i,steps,output:result.slice()});
 }
 return {output,stats,rows};
}
function mm(a,b){return a.map(row=>b[0].map((_,j)=>row.reduce((s,x,k)=>s+x*b[k][j],0)));}
function softmax(row){const max=Math.max(...row),e=row.map(x=>Math.exp(x-max)),sum=e.reduce((s,x)=>s+x,0);return e.map(x=>x/sum);}
function stack(tokens,p,block,order,trace,dense=false){
 const g=p.groups,d=p.dim,hd=d/p.heads;
 let x=tokens.map((t,r)=>g.embed[t].map((v,c)=>v+(c%2?Math.cos(r/10000**(2*Math.floor(c/2)/d)):Math.sin(r/10000**(2*Math.floor(c/2)/d)))));
 const embedded=trace?x.map(r=>r.slice()):[],layers=[],stats={score_evaluations:0,blocks:0,peak_score_buffer:0,score_matrix_slots:0,probability_matrix_slots:0,numerator_slots:hd};
 for(let li=0;li<p.layers;li++){
  const outputs=[],heads=[];
  for(let h=0;h<p.heads;h++){
   const [q,k,v]=['wq','wk','wv'].map(n=>mm(x,g[`l${li}_${n}${h}`]));let result;
   if(dense){
    const a=q.map((row,i)=>{const score=k.slice(0,i+1).map(key=>row.reduce((s,z,c)=>s+z*key[c],0)/Math.sqrt(hd));return softmax(score).concat(Array(x.length-i-1).fill(0));});
    result={output:mm(a,v),attention:a};
   }else{
    result=attention(q,k,v,block,order,trace);for(const key of ['score_evaluations','blocks'])stats[key]+=result.stats[key];stats.peak_score_buffer=Math.max(stats.peak_score_buffer,result.stats.peak_score_buffer);
   }
   outputs.push(result.output);if(trace)heads.push({q,k,v,...result});
  }
  const concat=tokens.map((_,r)=>outputs.flatMap(o=>o[r])),projected=mm(concat,g[`l${li}_wo`]),residual=x.map((row,r)=>row.map((z,c)=>z+projected[r][c]));
  if(trace)layers.push({input:x,heads,concat,projected,residual});x=residual;
 }
 const logits=mm(x,g.wout).map(row=>row.map((z,c)=>z+g.bout[0][c]));
 return {embedded,layers,logits,...(dense?{}:{stats})};
}
function validate(spec){
 exact(spec,['schema','model','tokens','target','block_size','order']);if(spec.schema!==SCHEMA)throw Error('Unsupported online schema');
 Legacy.model(spec.model);const p=spec.model;if(!Array.isArray(spec.tokens)||spec.tokens.length<1||spec.tokens.length>24)throw Error('tokens must contain 1..24 integers');
 for(const t of spec.tokens)integer(t,0,p.vocab-1,'token');integer(spec.target,0,p.vocab-1,'target');options(spec.block_size,spec.order);return p;
}
function summary(logits,target){const row=logits.at(-1),probabilities=softmax(row),prediction=row.indexOf(Math.max(...row));return {prediction,correct:prediction===target,loss:Legacy.ce(row,target),probabilities,target_probability:probabilities[target]};}
function forward_online(spec){const p=validate(spec);return stack(spec.tokens,p,spec.block_size,spec.order,false);}
function audit(spec){
 const p=validate(spec),online=stack(spec.tokens,p,spec.block_size,spec.order,true),dense=stack(spec.tokens,p,1,'forward',true,true),legacy=Legacy.run(spec.tokens,spec.target,p);
 let abs=0,scaled=0,old=0;for(let r=0;r<spec.tokens.length;r++)for(let c=0;c<p.vocab;c++){const a=online.logits[r][c],b=dense.logits[r][c],z=legacy.logits[r][c];abs=Math.max(abs,Math.abs(a-b));scaled=Math.max(scaled,Math.abs(a-b)/Math.max(1,Math.abs(a),Math.abs(b)));old=Math.max(old,Math.abs(z-b)/Math.max(1,Math.abs(z),Math.abs(b)));}
 return {schema:SCHEMA,tokens:spec.tokens.slice(),target:spec.target,block_size:spec.block_size,order:spec.order,online,dense,online_summary:summary(online.logits,spec.target),dense_summary:summary(dense.logits,spec.target),max_absolute_logit_error:abs,max_scaled_logit_error:scaled,legacy_max_scaled_logit_error:old,legacy_prediction:legacy.prediction};
}
const api={SCHEMA,attention,audit,forward_online};if(typeof module!=='undefined')module.exports=api;else root.OnlineEngine=api;
})(typeof window!=='undefined'?window:globalThis);

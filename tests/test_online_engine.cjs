'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),{execFileSync}=require('node:child_process'),O=require('../docs/online-engine.js'),data=require('../docs/data.json');
let numbers=0,steps=0,rows=0;
function compare(a,b,path='root'){
 assert.equal(typeof a,typeof b,path);
 if(typeof a==='number'){numbers++;assert.ok(Number.isFinite(b)&&Math.abs(a-b)<=1e-10*Math.max(1,Math.abs(a),Math.abs(b)),path+': '+a+' vs '+b);}
 else if(a===null||typeof a!=='object')assert.equal(a,b,path);
 else {assert.deepEqual(Object.keys(a).sort(),Object.keys(b).sort(),path);for(const key of Object.keys(a))compare(a[key],b[key],path+'.'+key);}
}
function spec(){const c=data.cases[0];return {schema:O.SCHEMA,model:structuredClone(data.models[0].model),tokens:c.tokens.slice(),target:c.target,block_size:3,order:'reverse'};}
test('all native model tensors and accumulator states, seeded real models and standalone kernels',()=>{
 const lines=execFileSync(process.env.PYTHON||'python',['tests/native_online_reports.py'],{encoding:'utf8',maxBuffer:256e6}).trim().split('\n');let models=0,kernels=0;
 for(const line of lines){const item=JSON.parse(line);let got;
  if(item.kind==='model'){models++;got=O.audit(item.spec);compare(item.report,got);compare(O.forward_online(item.spec),{...got.online,embedded:[],layers:[]});for(const l of got.online.layers)for(const h of l.heads)for(const r of h.rows){rows++;steps+=r.steps.length;}}
  else{kernels++;const a=item.args;got=O.attention(a.q,a.k,a.v,a.block_size,a.order,true);compare(item.report,got);for(const r of got.rows){rows++;steps+=r.steps.length;}}
 }
 assert.equal(models,106);assert.equal(kernels,80);console.log(JSON.stringify({full_model_reports:models,standalone_reports:kernels,query_rows:rows,accumulator_steps:steps,numeric_comparisons:numbers}));
});
test('strict validation, causal suffix, buffer counts, no mutation, extreme scores',()=>{
 const x=[[1,2],[3,4]],valid=[x,x,x];
 for(const bad of [[],[1],[[]],[[1],[2,3]],[[true]],[[NaN]],[[Infinity]],[[1e101]],Array(25).fill([1]),[Array(17).fill(1)]])for(let i=0;i<3;i++){const args=valid.slice();args[i]=bad;assert.throws(()=>O.attention(...args));}
 for(const [block,order,trace] of [[true,'forward',false],[0,'forward',false],[25,'forward',false],[1.5,'forward',false],[1,null,false],[1,'forward',1]])assert.throws(()=>O.attention(x,x,x,block,order,trace));
 for(const change of [{schema:1},{target:true},{tokens:[]},{tokens:[true]},{tokens:[12]},{block_size:0},{order:'shuffle'},{extra:1},{model:{}}])assert.throws(()=>O.audit({...spec(),...change}));
 const s=spec(),before=JSON.stringify(s),r=O.audit(s);assert.equal(JSON.stringify(s),before);assert.equal(r.online.stats.peak_score_buffer,3);
 for(let n=1;n<=24;n++)for(let b=1;b<=24;b++)for(const order of ['forward','reverse']){const q=Array.from({length:n},(_,i)=>[i/10,1]),k=q,v=q.map(r=>[-r[0],r[1]]),a=O.attention(q,k,v,b,order);assert.equal(a.stats.peak_score_buffer,Math.min(n,b));assert.equal(a.stats.score_evaluations,n*(n+1)/2);assert.equal(a.stats.blocks,Array.from({length:n},(_,i)=>Math.ceil((i+1)/b)).reduce((a,b)=>a+b,0));assert.deepEqual(a.rows,[]);for(let end=1;end<n;end++)assert.deepEqual(a.output.slice(0,end),O.attention(q.slice(0,end),k.slice(0,end),v.slice(0,end),b,order).output);}
 assert.deepEqual(O.attention([[1e100],[1e100]],[[-1e100],[1e100]],[[-1e100],[1e100]],1,'reverse').output,[[-1e100],[1e100]]);
});

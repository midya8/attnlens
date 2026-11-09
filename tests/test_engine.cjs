'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),{execFileSync}=require('node:child_process'),E=require('../docs/engine.js'),data=require('../docs/data.json');
let numeric=0;
function compare(a,b,path='root'){assert.equal(typeof a,typeof b,path);if(typeof a==='number'){numeric++;assert.ok(Number.isFinite(b)&&Math.abs(a-b)<=1e-10*Math.max(1,Math.abs(a),Math.abs(b)),`${path}: ${a} vs ${b}`);}else if(a===null||typeof a!=='object')assert.equal(a,b,path);else{assert.deepEqual(Object.keys(a).sort(),Object.keys(b).sort(),path);for(const k of Object.keys(a))compare(a[k],b[k],path+'.'+k);}}
test('all native tensor reports, all subsets, counterfactuals and held-out predictions',()=>{
 const native=JSON.parse(execFileSync(process.env.PYTHON||'python',['tests/native_reports.py'],{encoding:'utf8',env:{...process.env,PYTHONPATH:'.'},maxBuffer:80e6}));
 for(const item of native){const p=data.models.find(m=>m.id===item.model).model,c=data.cases.find(c=>c.id===item.case);for(const r of item.reports)compare(r,E.run(c.tokens,c.target,p,r.muted));compare(item.audit,E.inspect(c.tokens,c.target,p,[[0,1]]));}
 for(const m of data.models){const loss=Array(16).fill(0),correct=Array(16).fill(0);for(const c of data.evaluation.examples)for(const r of E.sweep(c.tokens,c.target,m.model)){loss[r.mask]+=r.loss;correct[r.mask]+=Number(r.correct);}for(let mask=0;mask<16;mask++)compare(m.evaluation[mask],{mask,mean_loss:loss[mask]/128,accuracy:correct[mask]/128});for(const [i,c]of data.cases.entries())compare(m.pinned[i],E.summary(E.run(c.tokens,c.target,m.model)));}
 console.log(JSON.stringify({models:2,fixtures:11,full_tensor_reports:352,full_audits:22,heldout_predictions:4096,numeric_comparisons:numeric}));
});
test('strict inputs, all-off residual, downstream recomputation, no mutation, stable extreme loss',()=>{
 const p=structuredClone(data.models[0].model),c=data.cases[0],before=JSON.stringify(p);
 for(const tokens of [[],[true],[12],[-1],[1.5]])assert.throws(()=>E.run(tokens,0,p));
 for(const muted of [[[0,0],[0,0]],[[2,0]],[[0,2]],[[true,0]],[[0]]])assert.throws(()=>E.run(c.tokens,c.target,p,muted));
 for(const change of [{dim:true},{schema:true},{heads:0},{layers:4},{extra:1}])assert.throws(()=>E.model({...p,...change}));
 const bad=structuredClone(p);bad.groups.embed[0][0]=Infinity;assert.throws(()=>E.model(bad));
 const all=E.run(c.tokens,c.target,p,[[0,0],[0,1],[1,0],[1,1]]);for(const l of all.layers)assert.deepEqual(l.residual,all.embedded);
 const b=E.run(c.tokens,c.target,p),a=E.run(c.tokens,c.target,p,[[0,1]]);assert.deepEqual(a.layers[0].attention,b.layers[0].attention);assert.notDeepEqual(a.layers[1].attention,b.layers[1].attention);
 assert.equal(JSON.stringify(p),before);assert.equal(E.ce([1000,-1000],1),2000);
 for(const tokens of [[1,2,11],[10,10,2,11],[1,2,10,11],[10,2,11,11],[10,true,1,11]])assert.throws(()=>E.counterfactuals(tokens));
});

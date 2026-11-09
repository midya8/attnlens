import copy
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from attnlens.model import Cache, Params, forward, cross_entropy, loss_and_grads
from attnlens.intervene import dump_model, load_model, run, sweep, counterfactuals, inspect


class TestInterventions(unittest.TestCase):
    def setUp(self):
        self.p=Params(12,4,2,2,random.Random(3))
        self.tokens=[2,10,3,4,5,6,7,11]

    def test_noop_matches_original_all_activations(self):
        cache=Cache();logits=forward(self.tokens,self.p,cache)
        report=run(self.tokens,3,self.p)
        self.assertEqual(report['logits'],logits)
        for li,(old,new) in enumerate(zip(cache.layer_caches,report['layers'])):
            for name in ('q','k','v','concat'):
                self.assertEqual(old[name],new[name])
            self.assertEqual(old['attn'],new['attention'])
            self.assertEqual(cache.layer_inputs[li],new['input'])

    def test_all_heads_off_is_embedding_residual(self):
        muted=[[li,h] for li in range(2) for h in range(2)]
        r=run(self.tokens,3,self.p,muted)
        for layer in r['layers']:
            self.assertEqual(layer['input'],r['embedded'])
            self.assertEqual(layer['residual'],r['embedded'])
            self.assertTrue(all(x==0 for row in layer['projected'] for x in row))

    def test_one_head_zeroes_exact_slice_and_recomputes_downstream(self):
        base=run(self.tokens,3,self.p)
        ablated=run(self.tokens,3,self.p,[[0,1]])
        for a,b in zip(ablated['layers'][0]['effective'],base['layers'][0]['concat']):
            self.assertEqual(a[:2],b[:2]);self.assertEqual(a[2:],[0,0])
        self.assertEqual(ablated['layers'][0]['attention'],base['layers'][0]['attention'])
        self.assertNotEqual(ablated['layers'][1]['attention'],base['layers'][1]['attention'])
        self.assertNotEqual(ablated['logits'],base['logits'])

    def test_no_mutation(self):
        before=dump_model(self.p);tokens=self.tokens[:]
        run(tokens,3,self.p,[[0,0]])
        self.assertEqual(before,dump_model(self.p));self.assertEqual(tokens,self.tokens)

    def test_checkpoint_roundtrip_full(self):
        loaded=load_model(json.loads(json.dumps(dump_model(self.p))))
        self.assertEqual(run(self.tokens,3,self.p),run(self.tokens,3,loaded))

    def test_strict_model_fields_shapes_values(self):
        original=dump_model(self.p)
        bad=[]
        for name,value in [('dim',True),('heads',0),('layers',4),('vocab',33),('schema',True)]:
            d=copy.deepcopy(original);d[name]=value;bad.append(d)
        d=copy.deepcopy(original);d['groups']['embed'][0][0]=float('nan');bad.append(d)
        d=copy.deepcopy(original);d['groups']['bout'][0].append(0);bad.append(d)
        d=copy.deepcopy(original);d['x']=1;bad.append(d)
        d=copy.deepcopy(original);d['groups'].pop('wout');bad.append(d)
        d=copy.deepcopy(original);d['groups']['embed'][0][0]=True;bad.append(d)
        for d in bad:
            with self.subTest(data=d),self.assertRaises(ValueError):load_model(d)

    def test_invalid_tokens_target_mute(self):
        for tokens,target,muted in [([],3,[]),([True],3,[]),([12],3,[]),([-1],3,[]),([1],True,[]),([1],3,[[0,0],[0,0]]),([1],3,[[2,0]]),([1],3,[[0,2]]),([1],3,[[False,0]]),([1],3,[[0]])]:
            with self.subTest(tokens=tokens,muted=muted),self.assertRaises(ValueError):run(tokens,target,self.p,muted)

    def test_all_subsets_including_noop_and_alloff(self):
        rows=sweep(self.tokens,3,self.p)
        self.assertEqual(len(rows),16);self.assertEqual(rows[0]['loss_delta'],0)
        self.assertEqual(len(rows[-1]['muted']),4)
        self.assertEqual({r['mask'] for r in rows},set(range(16)))
        for r in rows:
            self.assertEqual(r['loss_delta'],r['loss']-rows[0]['loss'])

    def test_counterfactuals_single_changed_position_and_correct_labels(self):
        rows=counterfactuals(self.tokens)
        self.assertEqual(len(rows),15)
        self.assertEqual([r['target'] for r in rows[:10]],list(range(10)))
        for r in rows:
            self.assertEqual(r['tokens'][r['tokens'].index(10)+1],r['target'])
            self.assertTrue(all(a==b for i,(a,b) in enumerate(zip(r['tokens'],self.tokens)) if i!=r['position']))
        for tokens in ([1,2,11],[10,10,2,11],[1,2,10,11],[10,2,11,11],[10,True,1,11]):
            with self.assertRaises(ValueError):counterfactuals(tokens)

    def test_future_change_causality_with_ablations(self):
        a=run(self.tokens,3,self.p,[[0,1]])
        tokens=self.tokens[:];tokens[-1]=0
        b=run(tokens,3,self.p,[[0,1]])
        self.assertEqual(a['logits'][:-1],b['logits'][:-1])
        for layer in a['layers']:
            for head in layer['attention']:
                for i,row in enumerate(head):self.assertTrue(all(x==0 for x in row[i+1:]))

    def test_cli_export_and_failure_no_partial(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'model.json';path.write_text(json.dumps(dump_model(self.p)))
            command=[sys.executable,'-m','attnlens.intervene','--model',str(path),'--tokens',','.join(map(str,self.tokens)),'--mute','0:1']
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout),inspect(self.tokens,3,self.p,[[0,1]]))
            path.write_text('{"schema":1,"schema":1}')
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,2);self.assertEqual(result.stdout,'')


class TestStableLoss(unittest.TestCase):
    def test_extreme_loss_is_not_clipped_and_shift_invariant(self):
        self.assertAlmostEqual(cross_entropy([1000,-1000],1),2000)
        self.assertAlmostEqual(cross_entropy([0,-2000],1),2000)
        self.assertEqual(cross_entropy([1000,-1000],0),0)

    def test_extreme_loss_matches_nonzero_gradient_finite_difference(self):
        p=Params(2,2,1,1,random.Random(1));p.bout[:]=[100,-100]
        tokens=[0,1];target=1
        loss,grads=loss_and_grads(tokens,target,p)
        self.assertGreater(loss,100)
        for i in range(2):
            old=p.bout[i];eps=1e-4
            p.bout[i]=old+eps;plus=loss_and_grads(tokens,target,p)[0]
            p.bout[i]=old-eps;minus=loss_and_grads(tokens,target,p)[0]
            p.bout[i]=old
            self.assertAlmostEqual((plus-minus)/(2*eps),grads['bout'][0][i],places=7)

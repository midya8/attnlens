# attnlens

A tiny attention-only transformer (two layers, two heads each, no MLP) trained in plain Python on a toy recall task, so I could look at what each head is doing and what happens when you switch one off. No autograd library, the backward pass is written out by hand.

The task: a sequence contains a marker token followed by a key, and at the end the model has to output that key. It's the classic setup where an induction-style head should show up.

```bash
python -m attnlens train --dim 8 --heads 2 --layers 2 --epochs 100 --seed 7
python -m attnlens heatmap --dim 8 --heads 2 --layers 2 --epochs 100 --seed 7
```

Training takes about 20 seconds on a laptop. With seed 7 the final loss is about 0.0003, accuracy is 100%, and you can see the second layer doing the work:

```
layer0 head0: entropy=1.182 prev=0.283 recall=0.041
layer0 head1: entropy=1.255 prev=0.243 recall=0.164
layer1 head0: entropy=0.781 prev=0.169 recall=0.803
layer1 head1: entropy=0.378 prev=0.163 recall=0.999
```

## Muting heads

https://midya8.github.io/attnlens/ loads the trained weights in the browser and lets you zero out any combination of heads and see the logits change. With all heads on, the checkpoint gets 128/128 held-out examples right. With all four off, only the embedding/residual path is left and it drops to 19/128. The other 14 combinations are in there too, and they don't add up neatly.

From the command line:

```bash
python -m attnlens.intervene --model attnlens-checkpoint.json \
  --tokens 6,10,2,8,2,7,5,11 --mute 0:1 > audit.json
```

Keep in mind that zeroing a head can push the model somewhere it never saw in training, so "accuracy dropped" isn't the same as "this head is important". INTERVENTIONS.md has more on that.

## Online softmax

A later addition: the same checkpoint run through block-by-block online softmax (the trick FlashAttention uses), compared against plain dense attention. `python -m attnlens.online docs/online-example.json --output audit.json`, or the page at `online.html`. Details in ONLINE.md. This is CPU Python, so no speed claims.

## Tests

```bash
python -m unittest discover -s tests   # 80 tests
node --test tests/*.cjs                # browser checks, needs Chrome
```

It's a toy model on synthetic data and one seed. Fun to poke at, not research.

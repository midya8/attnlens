"""Reproduce checkpoints from the unchanged 100-epoch training configuration."""
import json
import random
from pathlib import Path
from attnlens.model import Params
from attnlens.train import train
from attnlens.intervene import dump_model, run, sweep, summary, counterfactuals
from attnlens.data import make_dataset


def build():
    result = train(dim=8, heads=2, layers=2, epochs=100, lr=0.1, seed=7)
    initialized = Params(12, 8, 2, 2, random.Random(7))
    models = [dict(id='trained', label='Trained: 100 epochs, seed 7', model=dump_model(result['params'])),
              dict(id='initial', label='Initialization: same seed and architecture', model=dump_model(initialized))]
    cases = []
    for i, (tokens, target) in enumerate(make_dataset(8, 700011)):
        cases.append(dict(id=f'case-{i}', tokens=tokens, target=target))
    # Deliberate repeated-key and edge-marker structures, never selected by quality.
    for i, tokens in enumerate(([10,3,3,3,3,3,3,11], [1,1,1,1,10,2,1,11], [1,10,4,4,4,4,4,11])):
        cases.append(dict(id=f'edge-{i}', tokens=tokens, target=tokens[tokens.index(10)+1]))
    evaluation = make_dataset(128, 100007)
    for row in models:
        p = result['params'] if row['id'] == 'trained' else initialized
        sums = [0.0] * 16
        correct = [0] * 16
        for tokens, target in evaluation:
            for item in sweep(tokens, target, p):
                mask = item['mask']
                sums[mask] += item['loss']
                correct[mask] += item['correct']
        row['evaluation'] = [{'mask': k, 'mean_loss': sums[k]/128, 'accuracy': correct[k]/128} for k in range(16)]
        row['pinned'] = [summary(run(c['tokens'],c['target'],p)) for c in cases]
    return dict(schema=1, training={k:v for k,v in result.items() if k != 'params'},
                evaluation={'seed': 100007, 'size': 128, 'selection': 'all original held-out examples', 'examples': [dict(tokens=t,target=y) for t,y in evaluation]},
                cases=cases, models=models)


if __name__ == '__main__':
    Path('docs/data.json').write_text(json.dumps(build(), sort_keys=True, separators=(',', ':'),allow_nan=False)+'\n')

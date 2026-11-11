"""Emit full independently generated reports, never browser input intermediates."""
import json
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attnlens.online import SCHEMA, attention, audit
from attnlens.model import Params
from attnlens.intervene import dump_model

payload = json.loads((Path(__file__).resolve().parents[1]/'docs/data.json').read_text())
for model in payload['models']:
    for case in payload['cases']:
        for block, order in ((1,'forward'), (3,'reverse'), (24,'reverse')):
            spec = {'schema': SCHEMA, 'model': model['model'], 'tokens': case['tokens'],
                    'target': case['target'], 'block_size': block, 'order': order}
            print(json.dumps({'kind':'model', 'spec':spec, 'report':audit(spec)}, allow_nan=False))
rng = random.Random(9761)
for i in range(40):
    heads = rng.randint(1, 4)
    dim = max(2, heads*rng.randint(1, 16//heads))
    p = Params(rng.randint(2,32), dim, heads, rng.randint(1,3), rng)
    spec = {'schema':SCHEMA, 'model':dump_model(p), 'tokens':[rng.randrange(p.vocab) for _ in range(rng.randint(1,24))],
            'target':rng.randrange(p.vocab), 'block_size':rng.randint(1,24), 'order':rng.choice(['forward','reverse'])}
    print(json.dumps({'kind':'model','spec':spec,'report':audit(spec)}, allow_nan=False))
for i in range(80):
    n, d, dv = rng.randint(1,24), rng.randint(1,16), rng.randint(1,16)
    args = {'q':[[rng.uniform(-8,8) for _ in range(d)] for _ in range(n)],
            'k':[[rng.uniform(-8,8) for _ in range(d)] for _ in range(n)],
            'v':[[rng.uniform(-10,10) for _ in range(dv)] for _ in range(n)],
            'block_size':rng.randint(1,24), 'order':rng.choice(['forward','reverse'])}
    print(json.dumps({'kind':'kernel','args':args,'report':attention(**args, trace=True)}, allow_nan=False))

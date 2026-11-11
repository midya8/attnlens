#!/usr/bin/env bash
set -euo pipefail
root=$(pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
python -m pip wheel --no-deps --wheel-dir "$scratch/wheels" .
python -m venv "$scratch/venv"
"$scratch/venv/bin/python" -m pip install --no-deps "$scratch"/wheels/*.whl
cp docs/online-example.json "$scratch/online-example.json"
cp docs/data.json "$scratch/data.json"
cd "$scratch"
"$scratch/venv/bin/python" - <<'PY'
import importlib
import json
from pathlib import Path
import sys
for name in ['model','train','data','patterns','intervene','online']:
    m=importlib.import_module('attnlens.'+name)
    assert 'site-packages' in m.__file__,m.__file__
from attnlens.intervene import load_model, run, sweep, inspect, dump_model
from attnlens.model import forward
payload=json.loads(Path('data.json').read_text())
count=0
for model in payload['models']:
    p=load_model(model['model'])
    assert dump_model(p)==model['model']
    for case,pin in zip(payload['cases'],model['pinned']):
        report=run(case['tokens'],case['target'],p)
        assert report['logits']==forward(case['tokens'],p)
        for k,v in pin.items():assert report[k]==v
        assert len(sweep(case['tokens'],case['target'],p))==16
        inspect(case['tokens'],case['target'],p,[[0,1]])
        count+=1
Path('checkpoint.json').write_text(json.dumps(payload['models'][0]['model']))
from attnlens.online import audit, forward_online
spec=json.loads(Path('online-example.json').read_text())
report=audit(spec)
assert report['online']['logits']==forward_online(spec['tokens'],load_model(spec['model']),spec['block_size'],spec['order'])['logits']
assert report['max_scaled_logit_error']<1e-10
print(f'PASS: six installed module origins, two complete checkpoint roundtrips, {count} original-forward equivalences, 352 interventions and 22 audits outside source tree')
PY
"$scratch/venv/bin/python" -m attnlens.intervene --model checkpoint.json --tokens 6,10,2,8,2,7,5,11 --mute 0:1 > audit.json
"$scratch/venv/bin/python" -m attnlens train --dim 8 --heads 2 --layers 2 --epochs 100 --seed 7 --max-final-loss 0.05 --min-accuracy 0.95 --min-recall-attention 0.7
"$scratch/venv/bin/python" -m attnlens heatmap --dim 8 --heads 2 --layers 2 --epochs 100 --seed 7

"$scratch/venv/bin/python" -m attnlens.online online-example.json --output online-audit.json
"$scratch/venv/bin/attnlens-online" online-example.json --output online-entry.json
"$scratch/venv/bin/python" - <<'PY'
import json
from pathlib import Path
assert json.loads(Path('online-audit.json').read_text())==json.loads(Path('online-entry.json').read_text())
print('PASS: both installed online CLI paths agree outside source tree')
PY

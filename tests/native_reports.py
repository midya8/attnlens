import json
from attnlens.intervene import load_model, run, inspect, sweep
from pathlib import Path

data=json.loads(Path('docs/data.json').read_text())
reports=[]
for model in data['models']:
    p=load_model(model['model'])
    for case in data['cases']:
        rows=[]
        for item in sweep(case['tokens'],case['target'],p):
            rows.append(run(case['tokens'],case['target'],p,item['muted']))
        reports.append({'model':model['id'],'case':case['id'],'reports':rows,
                        'audit':inspect(case['tokens'],case['target'],p,[[0,1]])})
print(json.dumps(reports,allow_nan=False))

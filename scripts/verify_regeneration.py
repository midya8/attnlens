"""Re-train and recompute every exported value; preserve checked-in bytes."""
import json
from pathlib import Path
from build_workbench import build


def compare(a,b,path='root'):
    if type(a) != type(b):raise AssertionError(path)
    if type(a) is float:
        if abs(a-b)>1e-12*max(1,abs(a),abs(b)):raise AssertionError(path)
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:compare(a[k],b[k],path+'.'+k)
    elif isinstance(a,list):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+f'[{i}]')
    else:assert a==b,path


if __name__=='__main__':
    path=Path('docs/data.json');before=path.read_bytes()
    compare(json.loads(before),build())
    assert path.read_bytes()==before
    print('PASS: complete weights, training losses, 128 held-out examples, all 32 aggregate subset rows and fixture pins regenerated at 1e-12 normalized tolerance; file unchanged')

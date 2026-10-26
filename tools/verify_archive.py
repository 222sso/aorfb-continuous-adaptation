"""Verify the release manifests using only the Python standard library."""
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import csv, hashlib, json

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def main():
    rows=list(csv.DictReader((ROOT/'SHA256_manifest.csv').open(encoding='utf-8-sig')))
    for r in rows:
        path=ROOT/r['path']
        assert path.is_file() and path.stat().st_size==int(r['bytes']) and sha(path)==r['sha256'],r['path']
    score_records=0
    for i in range(3,6):
        with ZipFile(ROOT/f'data/Additional_file_{i}.zip') as z:
            assert z.testzip() is None
            for n in z.namelist():
                p=PurePosixPath(n);assert not p.is_absolute() and '..' not in p.parts,n
                score_records+=n.endswith('.csv.gz')
    assert score_records==96
    print(json.dumps({'status':'PASS','manifest_files':len(rows),'scored_run_files':score_records,
                      'new_experiments':0,'public_deposit_made':False}))
if __name__=='__main__':main()

"""Check the four existing v66 archive downloads; no model calls or network access."""
from pathlib import Path
from zipfile import ZipFile
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archives-dir',type=Path,required=True);a=p.parse_args()
 rows=json.loads((ROOT/'DATA_ARCHIVES.json').read_text())['archives']
 for r in rows:
  path=a.archives_dir/r['file'];assert path.is_file(),str(path)
  assert path.stat().st_size==r['bytes'] and sha(path)==r['sha256'],r['file']
  with ZipFile(path) as z:assert z.testzip() is None,r['file']
 print(json.dumps({'status':'PASS','verified_archives':len(rows),'new_experiments':0}))
if __name__=='__main__':main()

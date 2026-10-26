"""Read-only audit of ONLY the current 50-candidate IEFPCM panel and 8 calibration jobs."""
from pathlib import Path
import csv,hashlib,json,re,mmap
from collections import Counter
import pandas as pd
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'ablation').is_dir());HERE=Path(__file__).resolve().parent
PANEL=ROOT/'Figures/Fig6_rank_agreement_review9/Figure6_candidate_data.csv'
ids=pd.read_csv(PANEL).candidate_id.tolist();assert len(ids)==50
jobs=[('candidate',i,(next(p for p in Path(__file__).resolve().parents if (p/'gaussian').is_dir())/'gaussian'/'beta0p25_reddb_new_scaffold_dft50')/i) for i in ids]
jobs += [('calibration',f'CAL_{i:03d}',(next(p for p in Path(__file__).resolve().parents if (p/'gaussian').is_dir())/'gaussian'/'reddb_calibration')/f'CAL_{i:03d}') for i in range(1,9)]
rows=[]
for group,molecule,directory in jobs:
 for stage in ['opt','spgas','spsol']:
  inp=directory/f'{molecule}_{stage}.gjf';out=directory/f'{molecule}_{stage}.out'
  s=inp.read_text();route=next(x.strip() for x in s.splitlines() if x.strip().startswith('#'))
  assert 'PBEPBE/6-31++G(d,p)' in route and 'SCF=XQC' in route and 'Int=Ultrafine' in route
  assert 'Freq' not in route
  if stage=='opt':assert re.search(r'\n0\s+1\s*\n',s)
  if stage=='spsol':assert 'SCRF=(IEFPCM,Solvent=Water)' in route
  if stage!='opt':assert 'Geom=AllCheck' in route and 'Guess=Read' in route
  with out.open('rb') as f:
   with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as blob:
    tail=blob[max(0,len(blob)-70000):]
    assert b'Normal termination of Gaussian 16' in tail,(molecule,stage)
    assert b'EM64W-G16RevC.01' in re.sub(rb'\s+',b'',tail),(molecule,stage)
    assert re.search(rb'Charge\s*=\s*0\s+Multiplicity\s*=\s*1',blob),(molecule,stage)
    eps=re.findall(rb'Eps=\s+([\d.]+)',blob) if stage=='spsol' else []
    conv={}
    for label in ['Maximum Force','RMS     Force','Maximum Displacement','RMS     Displacement']:
     pattern=label.encode()+rb'\s+([-\d.]+)\s+([-\d.]+)\s+(YES|NO)'
     matches=re.findall(pattern,blob)
     if matches:conv[label]=[x.decode() for x in matches[-1]]
    if stage=='opt':
     assert blob.find(b'Optimization completed')>=0
     assert len(conv)==4,(molecule,conv)
    rec={'group':group,'molecule_id':molecule,'stage':stage,'input_path':str(inp),'input_sha256':hashlib.sha256(inp.read_bytes()).hexdigest(),'output_path':str(out),'output_bytes':out.stat().st_size,'route':route,'normal_termination':True,'revision':'C.01','charge':0,'multiplicity':1,'epsilon':float(eps[-1]) if eps else None,'convergence':json.dumps(conv)}
    rows.append(rec)
print('Audited archived calculations',len(rows),flush=True)
frame=pd.DataFrame(rows);frame.to_csv(HERE/'dft_input_output_audit_174.csv',index=False)
summary={'new_calculations':0,'panel_molecules':50,'calibration_molecules':8,'existing_jobs':len(rows),'all_normal_termination':all(r['normal_termination'] for r in rows),'all_revision':'Gaussian 16 C.01','all_charge_multiplicity':[0,1],'route_counts':dict(Counter(r['route'] for r in rows)),'aqueous_epsilon_values':sorted({r['epsilon'] for r in rows if r['epsilon'] is not None}),'frequency_jobs_in_audited_protocol':0,'geometry':'Gas-phase optimization followed by gas and IEFPCM-water single points on the same optimized geometry.','descriptor':'Electronic E_water - E_gas, converted from Hartree to eV; no vibrational/thermal/standard-state/protonation terms in the archived extraction.','scope':'Only the current Figure 7 50-molecule panel and 8 calibration references. SMD and other candidate/strategy datasets were excluded.'}
(HERE/'dft_protocol_audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

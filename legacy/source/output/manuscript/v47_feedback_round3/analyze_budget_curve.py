"""Feedback 7: random-order rarefaction of ONLY the 27 original Table 1 runs.
No new generation; deduplication across all nine decoding cells within each run.
"""
from pathlib import Path
import hashlib,itertools,json,sys
import numpy as np
import pandas as pd
from scipy.special import gammaln
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'ablation').is_dir())
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
import reviewer7_multiseed_analysis as original
BUDGETS=np.r_[0,np.arange(100,9001,100)]
METRICS=['validity_pct','uniqueness_pct','novelty_pct','target_pass_pct','yield_raw_target_pct','yield_unique_target_pct','yield_unique_novel_target_pct']

def rarefaction(counts,n,budgets):
 counts=np.asarray(counts,dtype=int)
 freq,multiplicity=np.unique(counts,return_counts=True)
 assert (counts>0).all() and counts.sum()<=n
 out=[]
 for b in budgets:
  assert 0<=b<=n
  if b==0: out.append(0.);continue
  include=np.ones(len(freq),dtype=float)
  good=b<=n-freq
  f=freq[good]
  logq=gammaln(n-f+1)-gammaln(n-f-b+1)-gammaln(n+1)+gammaln(n-b+1)
  include[good]=-np.expm1(np.minimum(logq,0.))
  out.append(float(include@multiplicity))
 return np.asarray(out)

def main():
 # Independent exact enumeration validates the finite-population formula.
 toy=['A','A','B',None,None,None];toy_counts=[2,1]
 exact=[]
 for b in range(7):
  vals=[len({toy[i] for i in inds if toy[i] is not None}) for inds in itertools.combinations(range(6),b)]
  exact.append(np.mean(vals))
 assert np.allclose(rarefaction(toy_counts,6,np.arange(7)),exact,rtol=0,atol=1e-12)
 rows,integrity=original.load_runs()
 assert len(rows)==243000 and len(integrity)==27
 models=original.load_models()
 scored=original.score(rows,models)
 scored.to_csv(HERE/'scored_Table1_runs_27.csv.gz',index=False)
 corpus=pd.read_csv(original.ARTIFACTS/'reddb.csv.gz')
 from rdkit import Chem
 ref={Chem.MolToSmiles(Chem.MolFromSmiles(s),canonical=True,isomericSmiles=False) for s in corpus.loc[corpus.split=='train','canonical_smiles']}
 cells=[]
 for key,g in scored.groupby(['backbone','arm','training_seed','temperature','top_p'],sort=True):
  r=original.summarize_cell(g,ref).to_dict();r.update(dict(zip(['backbone','arm','training_seed','temperature','top_p'],key)));cells.append(r)
 cell=pd.DataFrame(cells)
 per_seed=cell.groupby(['backbone','arm','training_seed'])[METRICS].mean().reset_index()
 expected=pd.read_csv(ROOT/'output/reviewer7_multiseed/seed_metrics_27.csv')
 merged=per_seed.merge(expected,on=['backbone','arm','training_seed'],suffixes=['_new','_paper'],validate='one_to_one')
 err=max(float(np.abs(merged[m+'_new']-merged[m+'_paper']).max()) for m in METRICS)
 assert len(merged)==27 and err<1e-8,(len(merged),err)
 records=[];ends=[];hashes=[]
 for key,g in scored.groupby(['backbone','arm','training_seed'],sort=True):
  assert len(g)==9000
  valid=g[g.rdkit_valid.astype(bool)&g.flat_canonical.notna()]
  hits=valid[valid.pred_target.astype(bool)]
  counts=hits.flat_canonical.value_counts().to_numpy()
  curve=rarefaction(counts,9000,BUDGETS)
  assert (np.diff(curve)>=-1e-8).all()
  assert abs(curve[-1]-hits.flat_canonical.nunique())<1e-9
  assert (curve<=BUDGETS+1e-8).all()
  for b,u in zip(BUDGETS,curve):records.append(dict(backbone=key[0],arm=key[1],training_seed=key[2],attempt_budget=int(b),expected_cumulative_unique_targets=u))
  run_cells=cell.loc[(cell.backbone==key[0])&(cell.arm==key[1])&(cell.training_seed==key[2])]
  end=dict(backbone=key[0],arm=key[1],training_seed=key[2],n_attempts=9000,n_target_outputs=len(hits),n_unique_targets_global=hits.flat_canonical.nunique(),sum_within_cell_unique_targets=int(run_cells.n_unique_targets.sum()))
  assert end['n_unique_targets_global']<=end['sum_within_cell_unique_targets']
  ends.append(end)
 for r in integrity:
  p=ROOT/r['source_file'];hashes.append({**r,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 curves=pd.DataFrame(records)
 summary=curves.groupby(['backbone','arm','attempt_budget']).expected_cumulative_unique_targets.agg(['mean','std']).reset_index()
 curves.to_csv(HERE/'budget_curve_seed_values.csv',index=False)
 summary.to_csv(HERE/'budget_curve_mean_sd.csv',index=False)
 pd.DataFrame(ends).to_csv(HERE/'budget_endpoints_seed_values.csv',index=False)
 pd.DataFrame(hashes).to_csv(HERE/'source_run_manifest_27.csv',index=False)
 cell.to_csv(HERE/'Table1_reproduced_cells_243.csv',index=False)
 summary.loc[summary.attempt_budget.isin([100,1000,3000,6000,9000])].to_csv(HERE/'budget_landmarks_mean_sd.csv',index=False)
 # Report where the mean continuous-vs-hard ordering changes, at evaluated 100-attempt resolution.
 crossing=[]
 for backbone in ['TCN','LSTM','TCN-LSTM']:
  p=summary[summary.backbone==backbone].pivot(index='attempt_budget',columns='arm',values='mean')
  diff=p.soft-p.hard
  for j in range(1,len(p)):
   if diff.iloc[j-1]*diff.iloc[j]<0:
    crossing.append({'backbone':backbone,'between_attempts':[int(p.index[j-1]),int(p.index[j])],'interpretation':'crossing of conditional mean rarefaction curves, not a statistical threshold'})
 audit={'new_training_runs':0,'new_generation_attempts':0,'source_runs':27,'original_attempts_reanalyzed':243000,'seeds':[42,43,44],'budget_range':[100,9000],'step':100,'record_order':'Expected cumulative count under uniform random permutation of each run\'s fixed 9000-attempt pool; this avoids temperature/top-p block-order artifacts.','mixture':'Pool has exactly 1000 original attempts from each of the nine Table 1 decoding conditions. Smaller prefixes have this mixture in expectation.','formula':'E[U(b)] = sum_m [1 - C(N-f_m,b)/C(N,b)], N=9000; f_m counts valid predicted-target occurrences of canonical molecule m.','deduplication':'Across all decoding conditions within each run; non-isomeric canonical identity; each training seed analyzed separately.','bands':'Mean +/- sample SD of three fine-tuning-seed conditional expectations; not confidence intervals or new independent generation.','Table1_metric_max_abs_error_pp':err,'Table1_vs_curve':'Table 1 averages within-cell 1000-attempt unique yields; the cumulative endpoint deduplicates the entire 9000-attempt pool. These are distinct estimands.','formula_validation':'Matches exact enumeration of all subsets of a six-attempt toy population; actual curves are monotone, bounded by attempts, and end at exact observed global distinct-target counts.','mean_curve_crossings':crossing}
 (HERE/'budget_analysis_audit.json').write_text(json.dumps(audit,indent=2))
 print(summary[summary.attempt_budget.isin([100,1000,9000])].to_string(index=False));print(json.dumps(audit,indent=2))
if __name__=='__main__':main()

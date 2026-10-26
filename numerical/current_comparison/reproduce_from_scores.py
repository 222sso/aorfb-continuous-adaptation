"""Verify the published96-run metrics from saved scores only; no model fitting/inference."""
from pathlib import Path
import argparse,hashlib
import numpy as np,pandas as pd
from scipy import stats
HERE=Path(__file__).resolve().parent
M=['validity_pct','uniqueness_pct','novelty_pct','target_pass_pct','yield_raw_target_pct','yield_unique_target_pct','yield_unique_novel_target_pct']
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--scores-root',type=Path,default=HERE.parent.parent);a=parser.parse_args()
 index=pd.read_csv(HERE/'scored_file_index_96.csv');reference=set(pd.read_csv(HERE/'novelty_reference_30829.csv.gz').flat_canonical)
 assert len(index)==96 and len(reference)==30829
 cells=[];seeds=[]
 for row in index.itertuples():
  p=a.scores_root/row.relative_path;assert hashlib.sha256(p.read_bytes()).hexdigest()==row.sha256,str(p)
  f=pd.read_csv(p,low_memory=False);assert len(f)==9000
  assert set(f.backbone)=={row.backbone} and set(f.condition)=={row.condition} and set(f.training_seed)=={row.training_seed}
  valid=f.rdkit_valid.astype(bool)&f.canonical_smiles.notna();pred=valid&f.solv_mean_eV.le(-1.5)&f.lumo_mean_eV.le(-4.)
  assert np.array_equal(pred,f.pred_target.astype(bool))
  entries=[]
  for (t,pn),g in f.groupby(['temperature','top_p']):
   assert len(g)==1000
   v=g[g.rdkit_valid.astype(bool)&g.canonical_smiles.notna()];target=v[v.pred_target.astype(bool)];u=set(v.flat_canonical);tu=set(target.flat_canonical)
   vals=[len(v)/len(g)*100,len(u)/len(v)*100,len(u-reference)/len(u)*100,len(target)/len(v)*100,len(target)/len(g)*100,len(tu)/len(g)*100,len(tu-reference)/len(g)*100]
   entry={'backbone':row.backbone,'condition':row.condition,'training_seed':row.training_seed,'temperature':t,'top_p':pn,**dict(zip(M,vals))};cells.append(entry);entries.append(entry)
  assert len(entries)==9
  sr={'backbone':row.backbone,'condition':row.condition,'training_seed':row.training_seed,**pd.DataFrame(entries)[M].mean().to_dict(),'unique_predicted_targets_9000':f.loc[pred,'flat_canonical'].nunique()};seeds.append(sr)
 computed=pd.DataFrame(seeds).sort_values(['backbone','condition','training_seed']).reset_index(drop=True)
 supplied=pd.read_csv(HERE/'seed_metrics_96.csv').sort_values(['backbone','condition','training_seed']).reset_index(drop=True)
 assert np.allclose(computed[M+['unique_predicted_targets_9000']],supplied[M+['unique_predicted_targets_9000']],rtol=0,atol=1e-10)
 expected_cells=pd.read_csv(HERE/'cell_metrics_864.csv').sort_values(['backbone','condition','training_seed','temperature','top_p']).reset_index(drop=True)
 got_cells=pd.DataFrame(cells).sort_values(['backbone','condition','training_seed','temperature','top_p']).reset_index(drop=True)
 assert np.allclose(got_cells[M],expected_cells[M],rtol=0,atol=1e-10)
 saved=pd.read_csv(HERE/'primary_tests_3.csv').set_index('backbone');primary=M[-1]
 for b in ['LSTM','TCN','TCN-LSTM']:
  g=computed[computed.backbone.eq(b)];h=g[g.condition.eq('hard_fixed10')].sort_values('training_seed')[primary].to_numpy();c=g[g.condition.eq('continuous_targetmax10')].sort_values('training_seed')[primary].to_numpy();assert len(h)==len(c)==16
  delta=c-h;tt,pp=stats.ttest_rel(c,h);ci=stats.t.interval(1-.05/3,15,loc=delta.mean(),scale=delta.std(ddof=1)/4);r=saved.loc[b]
  assert np.allclose([delta.mean(),tt,pp,min(3*pp,1),*ci],[r.mean_difference_pp,r.t,r.p_unadjusted,r.p_bonferroni3,r.simultaneous_ci_lower_pp,r.simultaneous_ci_upper_pp],rtol=1e-9,atol=1e-10)
 print('PASS:96runs,864cells,864000savedattempts; all seven metrics,pool counts and three prespecified primary tests reproduced.')
if __name__=='__main__':main()

from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
from rdkit import Chem
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'ablation').is_dir())
HERE=Path(__file__).resolve().parent
PIPE=ROOT/'q1_comparison_experiments_260806/generality_replication_260816'
sys.path.insert(0,str(ROOT/'scripts'))
import reviewer7_multiseed_analysis as manuscript
OUT=HERE.parent
METRICS=['validity_pct','uniqueness_pct','novelty_pct','target_pass_pct','yield_raw_target_pct','yield_unique_target_pct','yield_unique_novel_target_pct']

def main():
 assert (HERE/'all_experiments_complete.json').exists(),'New experiments must be complete before analysis'
 frames=[];integrity=[]
 for condition in ['equal','hard','soft','sampling_only','loss_only','equal1529']:
  for seed in [42,43,44]:
   if condition in ['equal','hard','soft']:run=PIPE/'results/reddb'/condition/f'seed_{seed}'
   else:run=HERE/'runs'/condition/f'seed_{seed}'
   frame=pd.read_csv(run/'generated_molecules.csv.gz').drop(columns=['flat_canonical'],errors='ignore')
   counts=frame.groupby(['temperature','top_p']).size()
   assert len(frame)==9000 and len(counts)==9 and (counts==1000).all()
   frame['condition']=condition;frame['training_seed']=seed
   frames.append(frame)
   train=json.loads((run/'training_complete.json').read_text())
   generation=json.loads((run/'generation_complete.json').read_text())
   assert hashlib.sha256((run/train['checkpoint']).read_bytes()).hexdigest()==train['checkpoint_sha256']
   if condition in ['sampling_only','loss_only','equal1529']:
    assert train['protocol_sha256']==hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest()
    assert train['script_sha256']==hashlib.sha256((HERE/'run_case.py').read_bytes()).hexdigest()
    assert generation['checkpoint_sha256']==train['checkpoint_sha256']
   h=hashlib.sha256((run/'generated_molecules.csv.gz').read_bytes()).hexdigest()
   integrity.append({'condition':condition,'training_seed':seed,'source':str(run),'generated_sha256':h,'pretrain_sha256':train['pretrain_checkpoint_sha256'],'checkpoint_sha256':train['checkpoint_sha256'],'n_train':train['n_corpus_train_molecules'],'n_common_validation':train['n_common_validation'],'selected_epoch':train['selected_epoch'],'epochs_completed':train['epochs_completed'],'best_common_unit_val_loss':min(train['val_losses']),'subset_targets':train.get('n_subset_targets'),'subset_sampling_seed':train.get('subset_sampling_seed')})
 for name,sha in json.loads((HERE/'frozen_input_hashes.json').read_text()).items():
  assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,('Frozen input changed',name)
 assert len({r['pretrain_sha256'] for r in integrity})==1
 assert {r['n_common_validation'] for r in integrity}=={311}
 rows=pd.concat(frames,ignore_index=True)
 scored=manuscript.score(rows,manuscript.load_models())
 scored.to_csv(OUT/'scored_controls_and_manuscript_TCN.csv.gz',index=False)
 corpus=pd.read_csv(PIPE/'artifacts/reddb.csv.gz')
 ref=set(Chem.MolToSmiles(Chem.MolFromSmiles(s),canonical=True,isomericSmiles=False) for s in corpus.loc[corpus.split=='train','canonical_smiles'])
 assert len(ref)==30829
 cells=[]
 for key,frame in scored.groupby(['condition','training_seed','temperature','top_p'],sort=True):
  s=manuscript.summarize_cell(frame,ref).to_dict()
  s.update(dict(zip(['condition','training_seed','temperature','top_p'],key)));cells.append(s)
 cells=pd.DataFrame(cells)
 seeds=cells.groupby(['condition','training_seed'])[METRICS].mean().reset_index()
 summary=seeds.groupby('condition')[METRICS].agg(['mean','std']);summary.columns=[f'{m}_{s}' for m,s in summary.columns];summary=summary.reset_index()
 # Reproduce the currently reported Table 1 TCN reference results before interpreting any contrast.
 original=pd.read_csv(ROOT/'output/reviewer7_multiseed/seed_metrics_27.csv')
 original=original.loc[original.backbone=='TCN']
 errors=[]
 for _,r in original.iterrows():
  check=seeds.loc[(seeds.condition==r.arm)&(seeds.training_seed==r.training_seed)].iloc[0]
  for m in METRICS:
   delta=abs(float(check[m])-float(r[m]));errors.append(delta)
 assert max(errors)<1e-8,('Reference reproduction failed',max(errors))
 differences=[]
 for lhs,rhs in [('sampling_only','equal'),('loss_only','equal'),('soft','sampling_only'),('soft','loss_only'),('hard','equal1529'),('equal1529','equal'),('soft','hard')]:
  a=seeds.loc[seeds.condition==lhs].set_index('training_seed')
  b=seeds.loc[seeds.condition==rhs].set_index('training_seed')
  for m in METRICS:
   d=a[m]-b[m]
   differences.append({'contrast':lhs+' minus '+rhs,'metric':m,'mean_difference_pp':d.mean(),'sample_sd_difference_pp':d.std(ddof=1),'seed42_difference_pp':d.loc[42],'seed43_difference_pp':d.loc[43],'seed44_difference_pp':d.loc[44]})
 # Interaction is descriptive on the percentage-point scale; no small-n significance claim.
 for m in METRICS:
  x=seeds.pivot(index='training_seed',columns='condition',values=m)
  d=x.soft-x.sampling_only-x.loss_only+x.equal
  differences.append({'contrast':'soft - sampling_only - loss_only + equal','metric':m,'mean_difference_pp':d.mean(),'sample_sd_difference_pp':d.std(ddof=1),'seed42_difference_pp':d.loc[42],'seed43_difference_pp':d.loc[43],'seed44_difference_pp':d.loc[44]})
 cells.to_csv(OUT/'cell_metrics_162.csv',index=False);seeds.to_csv(OUT/'seed_metrics_18.csv',index=False);summary.to_csv(OUT/'control_summary_mean_sd.csv',index=False)
 pd.DataFrame(differences).to_csv(OUT/'paired_descriptive_contrasts.csv',index=False);pd.DataFrame(integrity).to_csv(OUT/'run_integrity_18.csv',index=False)
 audit={'runs':18,'new_runs':9,'new_attempts':81000,'existing_TCN_reference_runs':9,'reference_max_abs_metric_error_pp':max(errors),'common_pretrain_checkpoint_sha256':integrity[0]['pretrain_sha256'],'validation_molecules':311,'novelty_reference_molecules':len(ref),'deduplication':'non-isomeric canonical molecules within each 1000-attempt cell','aggregation':'nine-cell means within seed, then mean and sample SD of three seeds','inference':'descriptive paired contrasts, no new significance claim','scope':'TCN only. Existing reference runs are exactly the current manuscript Table 1; no unrelated experimental results imported.'}
 (OUT/'analysis_audit.json').write_text(json.dumps(audit,indent=2))
 print(summary.to_string(index=False));print(json.dumps(audit,indent=2))
if __name__=='__main__':main()

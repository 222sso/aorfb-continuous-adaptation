"""New experiments required only by v47 feedback 4–5; original runs are read-only."""
from pathlib import Path
import argparse, csv, hashlib, json, math, os, sys, time
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'ablation').is_dir())
HERE=Path(__file__).resolve().parent
PIPE=ROOT/'q1_comparison_experiments_260806/generality_replication_260816'
sys.path[:0]=[str(PIPE),str(PIPE.parent),str(ROOT)]
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import numpy as np
import pandas as pd
import tensorflow as tf
import train_case as original
import backbones
from ablation.config import CFG
from ablation.datasets import make_tf_dataset_with_unit_w,make_tf_dataset_with_w,mix_two_datasets,split_by_w
from ablation.models import compile_model
from ablation.sample import detokenize_generated
from batch_sampling_v2 import generate_smiles_batched
from rdkit import Chem,RDLogger
RDLogger.DisableLog('rdApp.*')

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--condition',required=True,choices=['sampling_only','loss_only','equal1529']);ap.add_argument('--seed',required=True,type=int,choices=[42,43,44]);ap.add_argument('--stage',required=True,choices=['train','generate']);a=ap.parse_args()
 cfg=CFG();cfg.SEED=a.seed
 original.set_seed(a.seed)
 protocol=json.loads((HERE/'protocol.json').read_text());protocol_hash=digest(HERE/'protocol.json')
 run=HERE/'runs'/a.condition/f'seed_{a.seed}';run.mkdir(parents=True,exist_ok=True)
 vocab=json.loads((PIPE/'artifacts/vocab.json').read_text());stoi={str(k):int(v) for k,v in vocab['stoi'].items()};itos={int(k):str(v) for k,v in vocab['itos'].items()}
 corpus=pd.read_csv(PIPE/'artifacts/reddb.csv.gz');chembl=pd.read_csv(PIPE/'artifacts/chembl.csv.gz')
 train=corpus.loc[corpus.split=='train'].copy().reset_index(drop=True);valid=corpus.loc[corpus.split=='valid'].copy().reset_index(drop=True)
 assert len(stoi)==40 and len(train)==30829 and len(valid)==311 and train.is_target.sum()==1529
 pre_meta=json.loads((PIPE/'artifacts/pretrain_complete.json').read_text());pre=PIPE/'artifacts'/pre_meta['checkpoint']
 baseline=json.loads((PIPE/'results/reddb/soft'/f'seed_{a.seed}'/'training_complete.json').read_text())
 assert digest(pre)==baseline['pretrain_checkpoint_sha256']
 if a.stage=='train':
  done=run/'training_complete.json'
  if done.exists():
   assert json.loads(done.read_text())['protocol_sha256']==protocol_hash
   print('SKIP completed training',run,flush=True);return
  selected_seed=None
  if a.condition=='equal1529':
   selected_seed=10000+a.seed
   train=train.sample(n=1529,random_state=selected_seed).sort_index().reset_index(drop=True)
   train.to_csv(run/'random_subset_manifest.csv',index=False)
  smiles=train.encoded_smiles.astype(str).tolist();weights=train.property_weight.to_numpy(float)
  assert all(len(s)+2<=128 for s in smiles)
  n_hi=n_lo=0
  if a.condition=='sampling_only':
   (hi_s,hi_w),(lo_s,lo_w)=split_by_w(smiles,weights,1.2)
   assert len(hi_s)==1701 and len(lo_s)==29128
   hi_ds,n_hi=make_tf_dataset_with_w(hi_s,[1.]*len(hi_s),stoi,cfg,128,32,True,a.seed+120)
   lo_ds,n_lo=make_tf_dataset_with_w(lo_s,[1.]*len(lo_s),stoi,cfg,128,32,True,a.seed+121)
   ds=mix_two_datasets(hi_ds,lo_ds,p_a=.9,seed=a.seed+122)
   n_train=n_hi+n_lo
  elif a.condition=='loss_only':
   ds,n_train=make_tf_dataset_with_w(smiles,weights,stoi,cfg,128,32,True,a.seed+120);ds=ds.repeat()
  else:
   ds,n_train=make_tf_dataset_with_unit_w(smiles,stoi,cfg,128,32,True,a.seed+120);ds=ds.repeat()
  cds,n_chembl=make_tf_dataset_with_unit_w(chembl.encoded_smiles.astype(str).tolist(),stoi,cfg,128,32,True,a.seed+14)
  vds,n_valid=make_tf_dataset_with_unit_w(valid.encoded_smiles.astype(str).tolist(),stoi,cfg,128,32,False,55)
  ds=original.deterministic(tf.data.Dataset.sample_from_datasets([original.deterministic(ds),original.deterministic(cds.repeat())],weights=[.5,.5],seed=a.seed+200).prefetch(tf.data.AUTOTUNE))
  vds=original.deterministic(vds)
  model=backbones.build('A_TCN',len(stoi),cfg);model.load_weights(pre);compile_model(model,1e-4)
  path=run/'best_common_unit_val.weights.h5'
  started=time.monotonic()
  history=model.fit(ds,validation_data=vds,steps_per_epoch=800,validation_steps=math.ceil(n_valid/32),epochs=10,callbacks=[tf.keras.callbacks.ModelCheckpoint(path,save_weights_only=True,save_best_only=True,monitor='val_loss',mode='min'),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True,mode='min'),tf.keras.callbacks.CSVLogger(run/'training_history.csv')],verbose=2)
  vals=[float(v) for v in history.history['val_loss']]
  meta={'condition':a.condition,'training_seed':a.seed,'backbone':'A_TCN','n_corpus_train_molecules':len(train),'n_corpus_encoded':n_train,'n_common_validation':n_valid,'n_chembl_encoded':n_chembl,'n_subset_targets':int(train.is_target.sum()),'subset_sampling_seed':selected_seed,'high_pool_probability':.9 if a.condition=='sampling_only' else None,'sampling_enabled':a.condition=='sampling_only','loss_weight_enabled':a.condition=='loss_only','n_high_pool':n_hi,'n_low_pool':n_lo,'selection_objective':'common unit-weight held-out validation loss','selected_epoch':int(np.argmin(vals)+1),'epochs_completed':len(vals),'val_losses':vals,'checkpoint':path.name,'checkpoint_sha256':digest(path),'pretrain_checkpoint_sha256':digest(pre),'protocol_sha256':protocol_hash,'script_sha256':digest(__file__),'wall_seconds':time.monotonic()-started}
  write_json(done,meta);print(json.dumps(meta,indent=2),flush=True)
 else:
  meta=json.loads((run/'training_complete.json').read_text());assert meta['protocol_sha256']==protocol_hash
  done=run/'generation_complete.json'
  if done.exists():
   assert json.loads(done.read_text())['protocol_sha256']==protocol_hash
   print('SKIP completed generation',run,flush=True);return
  model=backbones.build('A_TCN',len(stoi),cfg);model.load_weights(run/meta['checkpoint'])
  ref=set(Chem.MolToSmiles(Chem.MolFromSmiles(s),canonical=True,isomericSmiles=False) for s in corpus.loc[corpus.split=='train','canonical_smiles'])
  frames=[];seed_records=[];started=time.monotonic()
  for t in [.7,.9,1.1]:
   for p in [.75,.85,.95]:
    key=f'v47_feedback_04_05|{a.seed}|{t:.2f}|{p:.2f}'
    gen_seed=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%(2**31)
    # Same stable decoding seed across the three new controls in each training-seed/grid cell.
    seqs=detokenize_generated(generate_smiles_batched(model,itos,stoi,cfg,gen_n=1000,temperature=t,top_p=p,seed=gen_seed,batch_size=256),cfg)
    data=[]
    for i,s in enumerate(seqs):
     m=Chem.MolFromSmiles(s)
     canon=Chem.MolToSmiles(m,canonical=True,isomericSmiles=True) if m is not None else None
     flat=Chem.MolToSmiles(m,canonical=True,isomericSmiles=False) if m is not None else None
     data.append({'attempt_in_cell':i+1,'rdkit_valid':m is not None,'canonical_smiles':canon,'flat_canonical':flat,'novel_molecule':flat not in ref if flat else False})
    frame=pd.DataFrame(data);frame['temperature']=t;frame['top_p']=p;frame['generation_seed']=gen_seed;frames.append(frame)
    seed_records.append({'temperature':t,'top_p':p,'generation_seed':gen_seed})
    print(a.condition,a.seed,t,p,'valid',int(frame.rdkit_valid.sum()),flush=True)
  frame=pd.concat(frames,ignore_index=True);assert len(frame)==9000
  frame.to_csv(run/'generated_molecules.csv.gz',index=False)
  write_json(done,{'condition':a.condition,'training_seed':a.seed,'attempts':len(frame),'cells':9,'attempts_per_cell':1000,'seeds':seed_records,'protocol_sha256':protocol_hash,'checkpoint_sha256':meta['checkpoint_sha256'],'wall_seconds':time.monotonic()-started})
if __name__=='__main__':main()

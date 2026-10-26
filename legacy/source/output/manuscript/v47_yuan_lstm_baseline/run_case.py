"""Only the authorized target-subset LSTM baseline, with existing manuscript helpers."""
from pathlib import Path
import argparse, datetime, hashlib, json, math, os, sys, time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PIPE=ROOT/'q1_comparison_experiments_260806/generality_replication_260816'
sys.path[:0]=[str(PIPE), str(PIPE.parent), str(ROOT)]
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import numpy as np
import pandas as pd
import tensorflow as tf
import train_case as original
import backbones
from ablation.config import CFG
from ablation.datasets import make_tf_dataset_with_unit_w
from ablation.models import compile_model
from ablation.sample import detokenize_generated
from batch_sampling_v2 import generate_smiles_batched
from rdkit import Chem, RDLogger, rdBase
RDLogger.DisableLog('rdApp.*')

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');temp.replace(path)

class Progress(tf.keras.callbacks.Callback):
    def __init__(self,path): super().__init__();self.path=path
    def on_epoch_begin(self,epoch,logs=None): self.epoch=epoch+1
    def on_train_batch_end(self,batch,logs=None):
        if (batch+1)%100==0:
            write_json(self.path,{'epoch':self.epoch,'step':batch+1,'steps_per_epoch':800,
                                 'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 'training_loss':float(logs['loss'])})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True,choices=[42,43,44]);ap.add_argument('--stage',required=True,choices=['train','generate']);a=ap.parse_args()
    protocol_hash=digest(HERE/'protocol.json')
    frozen=json.loads((HERE/'frozen_input_hashes.json').read_text())
    for name,expected in frozen.items():
        if digest(ROOT/name)!=expected: raise RuntimeError('Frozen input changed: '+name)
    cfg=CFG();cfg.SEED=a.seed;original.set_seed(a.seed)
    run=HERE/'runs'/f'seed_{a.seed}';run.mkdir(parents=True,exist_ok=True)
    vocab=json.loads((PIPE/'artifacts/vocab.json').read_text());stoi={str(k):int(v) for k,v in vocab['stoi'].items()};itos={int(k):str(v) for k,v in vocab['itos'].items()}
    corpus=pd.read_csv(PIPE/'artifacts/reddb.csv.gz');train=corpus[corpus.split.eq('train') & corpus.is_target.eq(True)];valid=corpus[corpus.split.eq('valid')]
    assert len(train)==1529 and len(valid)==311 and len(stoi)==40
    assert all(len(s)+2<=128 for s in train.encoded_smiles)
    pre=PIPE/'artifacts/ckpt_pretrain_B_LSTM_union.weights.h5'
    if a.stage=='train':
        tf.config.experimental.enable_tensor_float_32_execution(False)
        done=run/'training_complete.json'
        if done.exists():
            meta=json.loads(done.read_text());assert meta['protocol_sha256']==protocol_hash
            assert digest(run/meta['checkpoint'])==meta['checkpoint_sha256'];print('SKIP',run,flush=True);return
        if (run/'training_history.csv').exists():
            raise RuntimeError('Partial training exists; do not silently restart or overwrite.')
        ds,n=make_tf_dataset_with_unit_w(train.encoded_smiles.astype(str).tolist(),stoi,cfg,128,32,True,a.seed+120)
        vds,nv=make_tf_dataset_with_unit_w(valid.encoded_smiles.astype(str).tolist(),stoi,cfg,128,32,False,55)
        ds=original.deterministic(ds.repeat());vds=original.deterministic(vds)
        assert (n,nv)==(1529,311)
        model=backbones.build('B_LSTM',40,cfg);assert model.count_params()==409640
        model.load_weights(pre);compile_model(model,1e-4)
        # Keras conservatively disables XLA for all potentially cuDNN-compatible
        # LSTMs, even when recurrent_dropout prevents the cuDNN path. Here the
        # unchanged recurrent_dropout=0.1 guarantees the ordinary RNN path.
        # Enable compilation of those same operations; never switch B_LSTM_fast.
        assert model.get_layer('lstm').recurrent_dropout == 0.1
        model.get_layer('lstm').supports_jit=True
        model.jit_compile=True
        assert model.jit_compile
        write_json(run/'environment.json',{'python':sys.version,'tensorflow':tf.__version__,'numpy':np.__version__,'pandas':pd.__version__,'rdkit':rdBase.rdkitVersion,'gpu':[str(x) for x in tf.config.list_physical_devices('GPU')], 'training_execution':'XLA compilation of unchanged B_LSTM graph; inference/scoring unchanged', 'jit_compile':bool(model.jit_compile), 'training_tf32_enabled':False, 'env':{k:os.environ.get(k) for k in ['PYTHONHASHSEED','GPU_MEM_LIMIT_MB','OMP_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS']}})
        checkpoint=run/'best_common_unit_val.weights.h5';started=time.monotonic()
        history=model.fit(ds,validation_data=vds,steps_per_epoch=800,validation_steps=math.ceil(nv/32),epochs=10,callbacks=[
            tf.keras.callbacks.ModelCheckpoint(checkpoint,save_weights_only=True,save_best_only=True,monitor='val_loss',mode='min'),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True,mode='min'),
            tf.keras.callbacks.CSVLogger(run/'training_history.csv'),Progress(run/'progress.json')],verbose=2)
        vals=[float(v) for v in history.history['val_loss']]
        write_json(done,{'condition':'yuan_target_subset_tl','backbone':'B_LSTM','training_seed':a.seed,
            'n_target_training':n,'n_common_validation':nv,'n_chembl_replay':0,'loss_weight':'unit non-padding token weights',
            'selected_epoch':int(np.argmin(vals)+1),'epochs_completed':len(vals),'val_losses':vals,
            'selection_objective':'common unit-weight held-out validation loss','checkpoint':checkpoint.name,
            'checkpoint_sha256':digest(checkpoint),'pretrain_checkpoint_sha256':digest(pre),
            'protocol_sha256':protocol_hash,'script_sha256':digest(__file__),'wall_seconds':time.monotonic()-started})
        print('TRAINING COMPLETE',run,flush=True)
    else:
        meta=json.loads((run/'training_complete.json').read_text());assert meta['protocol_sha256']==protocol_hash
        assert digest(run/meta['checkpoint'])==meta['checkpoint_sha256']
        done=run/'generation_complete.json'
        if done.exists():
            gm=json.loads(done.read_text());assert gm['protocol_sha256']==protocol_hash
            assert digest(run/'generated_molecules.csv.gz')==gm['output_sha256'];print('SKIP',run,flush=True);return
        model=backbones.build('B_LSTM',40,cfg);model.load_weights(run/meta['checkpoint'])
        ref={Chem.MolToSmiles(Chem.MolFromSmiles(s),canonical=True,isomericSmiles=False) for s in corpus.loc[corpus.split.eq('train'),'canonical_smiles']}
        frames=[];seeds=[];started=time.monotonic()
        for t in [.7,.9,1.1]:
            for p in [.75,.85,.95]:
                gen_seed=int(hashlib.sha256(f'v47_yuan_target_subset_tl|{a.seed}|{t:.2f}|{p:.2f}'.encode()).hexdigest()[:8],16)%(2**31)
                cell=run/f'cell_T{t:.2f}_p{p:.2f}.csv.gz'
                if cell.exists():
                    frame=pd.read_csv(cell);assert len(frame)==1000 and frame.generation_seed.eq(gen_seed).all()
                else:
                    seqs=detokenize_generated(generate_smiles_batched(model,itos,stoi,cfg,gen_n=1000,temperature=t,top_p=p,seed=gen_seed,batch_size=256,
                        progress=lambda n,total:print(f'GENERATING seed={a.seed} T={t} p={p} {n}/{total}',flush=True)),cfg)
                    rows=[]
                    for i,s in enumerate(seqs):
                        mol=Chem.MolFromSmiles(s)
                        canonical=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True) if mol is not None else None
                        flat=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=False) if mol is not None else None
                        rows.append({'attempt_in_cell':i+1,'rdkit_valid':mol is not None,'canonical_smiles':canonical,'novel_molecule':flat not in ref if flat else False})
                    frame=pd.DataFrame(rows);frame['temperature']=t;frame['top_p']=p;frame['generation_seed']=gen_seed
                    frame.to_csv(cell,index=False)
                frames.append(frame);seeds.append({'temperature':t,'top_p':p,'generation_seed':gen_seed})
                print(f'CELL COMPLETE seed={a.seed} T={t} p={p} valid={frame.rdkit_valid.sum()}',flush=True)
        frame=pd.concat(frames,ignore_index=True);assert len(frame)==9000
        frame.to_csv(run/'generated_molecules.csv.gz',index=False)
        write_json(done,{'condition':'yuan_target_subset_tl','training_seed':a.seed,'attempts':len(frame),
            'cells':9,'attempts_per_cell':1000,'decoding_seeds':seeds,'protocol_sha256':protocol_hash,
            'checkpoint_sha256':meta['checkpoint_sha256'],'output_sha256':digest(run/'generated_molecules.csv.gz'),
            'wall_seconds':time.monotonic()-started})
        print('GENERATION COMPLETE',run,flush=True)

if __name__=='__main__':main()

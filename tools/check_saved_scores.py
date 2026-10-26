"""Check the existing864000 scored attempts without predictors or hypothesis tests."""
from pathlib import Path
from zipfile import ZipFile
import gzip, hashlib, io, json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
NUM=ROOT/'numerical/current_comparison'
METRICS=['validity_pct','uniqueness_pct','novelty_pct','target_pass_pct',
         'yield_raw_target_pct','yield_unique_target_pct','yield_unique_novel_target_pct']
def main():
    index=pd.read_csv(NUM/'scored_file_index_96.csv')
    reference=set(pd.read_csv(NUM/'novelty_reference_30829.csv.gz').flat_canonical)
    assert len(index)==96 and len(reference)==30829
    zips={b:ZipFile(ROOT/f'data/Additional_file_{i}.zip') for i,b in enumerate(['LSTM','TCN','TCN-LSTM'],3)}
    cells=[];seeds=[];attempts=0
    try:
        for row in index.itertuples():
            blob=zips[row.backbone].read(row.relative_path)
            assert hashlib.sha256(blob).hexdigest()==row.sha256
            f=pd.read_csv(io.BytesIO(blob),compression='gzip',low_memory=False)
            assert len(f)==9000
            assert set(f.backbone)=={row.backbone} and set(f.condition)=={row.condition} and set(f.training_seed)=={row.training_seed}
            valid=f.rdkit_valid.astype(bool)&f.canonical_smiles.notna()
            pred=valid&f.solv_mean_eV.le(-1.5)&f.lumo_mean_eV.le(-4.)
            assert np.array_equal(pred,f.pred_target.astype(bool))
            entries=[]
            for (t,p),g in f.groupby(['temperature','top_p']):
                assert len(g)==1000
                v=g[g.rdkit_valid.astype(bool)&g.canonical_smiles.notna()]
                target=v[v.pred_target.astype(bool)];u=set(v.flat_canonical);tu=set(target.flat_canonical)
                values=[len(v)/1000*100,len(u)/len(v)*100,len(u-reference)/len(u)*100,
                        len(target)/len(v)*100,len(target)/1000*100,len(tu)/1000*100,len(tu-reference)/1000*100]
                record={'backbone':row.backbone,'condition':row.condition,'training_seed':row.training_seed,
                        'temperature':t,'top_p':p,**dict(zip(METRICS,values))}
                cells.append(record);entries.append(record)
            assert len(entries)==9
            seeds.append({'backbone':row.backbone,'condition':row.condition,'training_seed':row.training_seed,
                          **pd.DataFrame(entries)[METRICS].mean().to_dict(),
                          'unique_predicted_targets_9000':f.loc[pred,'flat_canonical'].nunique()})
            attempts+=len(f)
    finally:
        for z in zips.values():z.close()
    keys=['backbone','condition','training_seed']
    for actual,path,key,values in [(pd.DataFrame(seeds),NUM/'seed_metrics_96.csv',keys,METRICS+['unique_predicted_targets_9000']),
                                    (pd.DataFrame(cells),NUM/'cell_metrics_864.csv',keys+['temperature','top_p'],METRICS)]:
        a=actual.sort_values(key).reset_index(drop=True)
        b=pd.read_csv(path).sort_values(key).reset_index(drop=True)
        assert len(a)==len(b) and a[key].equals(b[key])
        assert np.allclose(a[values],b[values],rtol=0,atol=1e-10)
    assert attempts==864000
    print(json.dumps({'status':'PASS','runs':96,'cells':864,'saved_attempts':attempts,
                      'existing_metrics_checked':7,'pool_counts_checked':96,
                      'new_predictions':0,'new_hypothesis_tests':0,'new_experiments':0}))
if __name__=='__main__':main()

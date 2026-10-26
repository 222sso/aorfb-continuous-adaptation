"""Score only the new baseline and compare only the paper's existing LSTM arms."""
from pathlib import Path
import hashlib, importlib.util, json, sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy import stats
from rdkit import Chem
from scripts import reviewer7_multiseed_analysis as original

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    protocol=json.loads((HERE/'protocol.json').read_text());metrics=protocol['evaluation']['metrics']
    frozen=json.loads((HERE/'frozen_input_hashes.json').read_text())
    for name,expected in frozen.items():assert digest(ROOT/name)==expected,name
    frames=[];audit=[]
    for seed in [42,43,44]:
        run=HERE/'runs'/f'seed_{seed}'
        gm=json.loads((run/'generation_complete.json').read_text())
        tm=json.loads((run/'training_complete.json').read_text())
        assert gm['output_sha256']==digest(run/'generated_molecules.csv.gz')
        assert tm['protocol_sha256']==digest(HERE/'protocol.json')==gm['protocol_sha256']
        frame=pd.read_csv(run/'generated_molecules.csv.gz')
        counts=frame.groupby(['temperature','top_p']).size()
        assert len(frame)==9000 and len(counts)==9 and counts.eq(1000).all()
        frame['backbone']='LSTM';frame['arm']='yuan_target_subset_tl';frame['training_seed']=seed
        frames.append(frame);audit.append({'seed':seed,'selected_epoch':tm['selected_epoch'],'epochs_completed':tm['epochs_completed'],
            'n_attempts':len(frame),'n_valid':int(frame.rdkit_valid.sum()),'train_seconds':tm['wall_seconds'],
            'generation_seconds':gm['wall_seconds'],'checkpoint_sha256':tm['checkpoint_sha256'],
            'generated_sha256':gm['output_sha256']})
    rows=pd.concat(frames,ignore_index=True)
    scored=original.score(rows,original.load_models())
    scored.to_csv(HERE/'scored_yuan_baseline_27000.csv.gz',index=False)
    corpus=pd.read_csv(original.ARTIFACTS/'reddb.csv.gz')
    ref={Chem.MolToSmiles(Chem.MolFromSmiles(s),canonical=True,isomericSmiles=False) for s in corpus.loc[corpus.split.eq('train'),'canonical_smiles']}
    assert len(ref)==30829
    keys=['backbone','arm','training_seed','temperature','top_p']
    cells=scored.groupby(keys,sort=True).apply(lambda g:original.summarize_cell(g,ref),include_groups=False).reset_index()
    assert len(cells)==27
    cells.to_csv(HERE/'new_cell_metrics_27.csv',index=False)
    seeds=cells.groupby(['backbone','arm','training_seed'])[metrics].mean().reset_index()
    reference_cells=pd.read_csv(ROOT/'output/reviewer7_multiseed/cell_metrics_243.csv')
    reference_cells=reference_cells[reference_cells.backbone.eq('LSTM')].copy()
    reference_seeds=pd.read_csv(ROOT/'output/reviewer7_multiseed/seed_metrics_27.csv')
    reference_seeds=reference_seeds[reference_seeds.backbone.eq('LSTM')].copy()
    assert len(reference_cells)==81 and len(reference_seeds)==9
    check=reference_cells.groupby(['backbone','arm','training_seed'])[metrics].mean().reset_index()
    joined=check.merge(reference_seeds,on=['backbone','arm','training_seed'],suffixes=('_recomputed','_frozen'),validate='one_to_one')
    assert all(np.allclose(joined[m+'_recomputed'],joined[m+'_frozen'],rtol=0,atol=1e-10) for m in metrics)
    combined=pd.concat([reference_seeds,seeds],ignore_index=True)
    combined_cells=pd.concat([reference_cells,cells],ignore_index=True)
    combined.to_csv(HERE/'seed_metrics_12.csv',index=False)
    combined_cells.to_csv(HERE/'cell_metrics_108.csv',index=False)
    summary=combined.groupby(['backbone','arm'])[metrics].agg(['mean','std'])
    summary.columns=[f'{m}_{s}' for m,s in summary.columns];summary=summary.reset_index()
    summary.to_csv(HERE/'comparison_mean_sd.csv',index=False)
    display=summary[['backbone','arm']].copy()
    for m in metrics:display[m]=summary.apply(lambda r:f'{r[m+"_mean"]:.2f} ± {r[m+"_std"]:.2f}',axis=1)
    display.to_csv(HERE/'comparison_display.csv',index=False)
    tests=[]
    for arm in ['equal','hard','soft']:
        left=seeds.set_index('training_seed').loc[[42,43,44]]
        right=reference_seeds[reference_seeds.arm.eq(arm)].set_index('training_seed').loc[[42,43,44]]
        for m in metrics:
            d=left[m].to_numpy()-right[m].to_numpy();test=stats.ttest_1samp(d,0)
            tests.append({'contrast':f'yuan_target_subset_tl - {arm}','metric':m,'n_pairs':3,
                'mean_difference_pp':float(d.mean()),'sd_difference_pp':float(d.std(ddof=1)),
                'seed42_difference_pp':d[0],'seed43_difference_pp':d[1],'seed44_difference_pp':d[2],
                't_df2':float(test.statistic),'p_unadjusted':float(test.pvalue),
                'p_bonferroni_21':min(1.,float(test.pvalue)*21),
                'significant_bonferroni_0_05':bool(float(test.pvalue)*21<.05)})
    pd.DataFrame(tests).to_csv(HERE/'exploratory_paired_tests_21.csv',index=False)
    pd.DataFrame(audit).to_csv(HERE/'run_audit_3.csv',index=False)
    (HERE/'analysis_audit.json').write_text(json.dumps({'status':'complete','n_new_attempts':len(scored),
        'n_new_runs':3,'n_new_cells':len(cells),'n_reference_runs':9,'reference_backbone':'LSTM only',
        'n_common_novelty_reference':len(ref),'frozen_inputs_unchanged':True,'existing_seed_metrics_reproduced':True,
        'original_42_test_file_unchanged':True,'new_exploratory_test_family_n':len(tests),
        'all_new_training_and_generation_protocol_hashes_match':True,
        'protocol_sha256':digest(HERE/'protocol.json')},indent=2))
    print(display.to_string(index=False),flush=True)
    print('Analysis complete:',HERE,flush=True)

if __name__=='__main__':main()

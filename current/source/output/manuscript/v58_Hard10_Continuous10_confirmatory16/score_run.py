"""Score one authorized run with the unchanged frozen evaluator; no hypothesis tests."""
import argparse
import json
import sys
import numpy as np
import pandas as pd
from study import HERE, ROOT, PIPE, PROTOCOL, audit_generation, audit_score, digest, frozen_check, job_by_id, now, run_path, write_json

sys.path.insert(0, str(ROOT))
from scripts import reviewer7_multiseed_analysis as evaluator
from rdkit import Chem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    frozen_check()
    job = job_by_id(args.run_id)
    run = run_path(job)
    if (run / 'scoring_complete.json').exists():
        audit_score(job)
        return
    _, gm, rows = audit_generation(job)
    rows['backbone'] = job['backbone']
    rows['condition'] = job['condition']
    rows['training_seed'] = job['training_seed']
    rows['attempt_index_within_run'] = np.arange(9000)
    corpus = pd.read_csv(PIPE / 'artifacts/reddb.csv.gz')
    train = corpus[corpus.split.eq('train')]
    reference = {Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True, isomericSmiles=False)
                 for s in train.canonical_smiles}
    assert len(reference) == 30829
    scored = evaluator.score(rows, evaluator.load_models())
    keys = ['backbone', 'condition', 'training_seed', 'temperature', 'top_p']
    cells = scored.groupby(keys, sort=True).apply(
        lambda g: evaluator.summarize_cell(g, reference), include_groups=False).reset_index()
    assert len(scored) == 9000 and len(cells) == 9 and cells.n_attempts.eq(1000).all()
    primary = PROTOCOL['statistics']['primary_metric']
    assert np.isfinite(cells[primary]).all()
    seed_metrics = {m: float(cells[m].mean()) for m in PROTOCOL['evaluation']['metrics']}
    hits = scored[scored.rdkit_valid.astype(bool) & scored.pred_target.astype(bool)]
    seed_metrics.update({
        'run_id': job['run_id'], 'backbone': job['backbone'], 'condition': job['condition'],
        'training_seed': job['training_seed'], 'n_attempts': 9000,
        'unique_predicted_targets_9000': int(hits.flat_canonical.nunique())})
    scored.to_csv(run / 'scored_molecules.csv.gz', index=False)
    cells.to_csv(run / 'cell_metrics_9.csv', index=False)
    write_json(run / 'seed_metrics.json', seed_metrics)
    frozen_check()
    write_json(run / 'scoring_complete.json', {
        'run_id': job['run_id'], 'n_attempts': 9000, 'n_cells': 9, 'completed_utc': now(),
        'generated_sha256': gm['generated_sha256'],
        'protocol_sha256': digest(HERE / 'protocol.json'),
        'scored_sha256': digest(run / 'scored_molecules.csv.gz'),
        'cells_sha256': digest(run / 'cell_metrics_9.csv'),
        'seed_sha256': digest(run / 'seed_metrics.json'), 'predictor_refitted': False})
    print('Scoring complete:', job['run_id'], flush=True)


if __name__ == '__main__':
    main()

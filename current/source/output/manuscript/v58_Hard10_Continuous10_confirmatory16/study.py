"""Integrity and path helpers for the fixed, 96-run validation only."""
from pathlib import Path
import datetime
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PIPE = ROOT / 'q1_comparison_experiments_260806/generality_replication_260816'
PYTHON = '/home/sohyun/miniconda3/envs/tf-gpu/bin/python'
PROTOCOL = json.loads((HERE / 'protocol.json').read_text())
JOBS = json.loads((HERE / 'jobs_96.json').read_text())


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def frozen_check():
    freeze = json.loads((HERE / 'EXECUTION_FROZEN.json').read_text())
    for relative, expected in freeze['study_files'].items():
        assert digest(HERE / relative) == expected, 'Changed study file: ' + relative
    for relative, expected in json.loads((HERE / 'frozen_input_hashes.json').read_text()).items():
        assert digest(ROOT / relative) == expected, 'Changed inherited input: ' + relative
    assert freeze['training_started_at_freeze'] is False


def job_by_id(run_id):
    matches = [j for j in JOBS if j['run_id'] == run_id]
    assert len(matches) == 1
    return matches[0]


def run_path(job):
    return HERE / job['run_relative']


def audit_training(job):
    import h5py
    import numpy as np
    import pandas as pd
    run = run_path(job)
    tm = json.loads((run / 'training_complete.json').read_text())
    assert tm['training_seed'] == job['training_seed']
    assert tm['dataset'] == 'reddb' and tm['arm'] == job['arm']
    assert tm['backbone'] == job['model']
    assert tm['selected_epoch'] == tm['epochs_completed'] == 10
    assert tm['optimizer_updates'] == 8000 and tm['steps_per_epoch'] == 800
    assert tm['reddb_mix_ratio'] == 1 and tm['n_common_validation'] == 311
    assert tm['protocol_sha256'] == digest(HERE / 'protocol.json')
    assert tm['pretrain_checkpoint_sha256'] == PROTOCOL['backbones'][job['backbone']]['sha256']
    assert tm['checkpoint_sha256'] == digest(run / tm['checkpoint'])
    with h5py.File(run / tm['checkpoint'], 'r') as h:
        assert int(h['optimizer/vars/0'][()]) == 8000
    history = pd.read_csv(run / 'training_history.csv')
    assert len(history) == 10 and history.epoch.tolist() == list(range(10))
    assert np.allclose(history.val_loss, tm['val_losses'], rtol=0, atol=1e-10)
    if job['arm'] == 'hard':
        assert tm['n_corpus_encoded'] == tm['n_corpus_train_molecules'] == 1529
        assert not tm['loss_weight_enabled']
    else:
        assert tm['n_corpus_encoded'] == tm['n_corpus_train_molecules'] == 30829
        assert tm['target_override_n'] == 1529 and tm['target_override_weight'] == 10
        assert tm['non_target_weights_unchanged'] and tm['loss_weight_enabled']
        assert tm['n_high_pool_encoded'] == 2045 and tm['n_low_pool_encoded'] == 28784
        assert tm['high_pool_probability'] == .9
        assert tm['training_weights_sha256'] == digest(HERE / 'training_weights_30829.csv.gz')
    return tm


def audit_generation(job):
    import pandas as pd
    tm = audit_training(job)
    run = run_path(job)
    gm = json.loads((run / 'generation_complete.json').read_text())
    assert gm['run_id'] == job['run_id'] and gm['total_attempts'] == 9000
    assert gm['checkpoint_sha256'] == tm['checkpoint_sha256']
    assert gm['protocol_sha256'] == digest(HERE / 'protocol.json')
    assert gm['decoding_seed_plan_sha256'] == digest(HERE / 'decoding_seed_plan.csv')
    assert gm['generated_sha256'] == digest(run / 'generated_molecules.csv.gz')
    rows = pd.read_csv(run / 'generated_molecules.csv.gz')
    plan = pd.read_csv(HERE / 'decoding_seed_plan.csv')
    plan = plan[plan.run_id.eq(job['run_id'])]
    assert len(rows) == 9000 and len(plan) == 9
    assert rows.groupby(['temperature', 'top_p']).size().eq(1000).all()
    for r in plan.itertuples():
        group = rows[rows.temperature.eq(r.temperature) & rows.top_p.eq(r.top_p)]
        assert len(group) == 1000 and group.decoding_seed.eq(r.decoding_seed).all()
    return tm, gm, rows


def audit_score(job):
    run = run_path(job)
    tm, gm, _ = audit_generation(job)
    sm = json.loads((run / 'scoring_complete.json').read_text())
    assert sm['run_id'] == job['run_id'] and sm['n_attempts'] == 9000
    assert sm['generated_sha256'] == gm['generated_sha256']
    assert sm['protocol_sha256'] == digest(HERE / 'protocol.json')
    assert sm['scored_sha256'] == digest(run / 'scored_molecules.csv.gz')
    assert sm['cells_sha256'] == digest(run / 'cell_metrics_9.csv')
    assert sm['seed_sha256'] == digest(run / 'seed_metrics.json')
    return tm, gm, sm

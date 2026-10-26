"""Original sampler, original per-arm RNG rule; new fixed seed manifest only."""
import argparse
import json
import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from study import HERE, ROOT, PIPE, audit_training, digest, frozen_check, job_by_id, now, run_path, write_json

sys.path[:0] = [str(PIPE), str(PIPE.parent), str(ROOT)]
import backbones
import train_case as original
from ablation.config import CFG
from ablation.sample import detokenize_generated
from ablation.vocab import encode_smiles_list
from batch_sampling_v2 import generate_smiles_batched
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    frozen_check()
    job = job_by_id(args.run_id)
    run = run_path(job)
    tm = audit_training(job)
    if (run / 'generation_complete.json').exists():
        from study import audit_generation
        audit_generation(job)
        return
    original.set_seed(job['training_seed'])
    cfg = CFG()
    vocab = json.loads((PIPE / 'artifacts/vocab.json').read_text())
    stoi = vocab['stoi']
    itos = {int(k): s for k, s in vocab['itos'].items()}
    model = backbones.build(job['model'], len(stoi), cfg)
    model.load_weights(run / tm['checkpoint'])
    # Exactly the execution-only parity/fallback procedure used in v52-v54.
    tf.config.optimizer.set_experimental_options({'remapping': False, 'arithmetic_optimization': False})

    @tf.function(input_signature=[tf.TensorSpec((None, 127), tf.int32)], jit_compile=False)
    def forward(inp):
        return model(inp, training=False)

    def call(inp, training=False):
        assert training is False
        return forward(tf.convert_to_tensor(inp, dtype=tf.int32))

    corpus = pd.read_csv(PIPE / 'artifacts/reddb.csv.gz')
    validation = corpus[corpus.split.eq('valid')]
    arr = encode_smiles_list(validation.encoded_smiles.astype(str).tolist(), stoi, cfg, 128)[:, :-1]
    identical, maximum, probes = True, 0., []
    for idx in [np.arange(256), np.arange(256, 311), np.arange(232)]:
        eager = model(arr[idx], training=False).numpy()
        graph = call(arr[idx]).numpy()
        delta = float(np.max(np.abs(eager - graph)))
        same = bool(np.array_equal(eager, graph))
        identical &= same
        maximum = max(maximum, delta)
        probes.append({'batch_n': len(idx), 'bitwise_equal': same, 'max_abs_difference': delta})
    write_json(run / 'selected_checkpoint_generation_parity.json', {
        'n_common_validation_inputs': 311, 'all_position_probabilities_bitwise_equal': identical,
        'maximum_absolute_difference': maximum, 'probes': probes,
        'used_native_eager_fallback': not identical, 'no_XLA': True, 'precision': 'original default'})
    sampler_model = call if identical else model
    plan = pd.read_csv(HERE / 'decoding_seed_plan.csv')
    plan = plan[plan.run_id.eq(job['run_id'])]
    assert len(plan) == 9
    frames = []
    started = time.monotonic()
    cells = run / 'cells'
    cells.mkdir(exist_ok=True)
    for row in plan.itertuples():
        file = cells / f'T{row.temperature:.2f}_p{row.top_p:.2f}.csv.gz'
        if file.exists():
            cell = pd.read_csv(file)
            assert len(cell) == 1000 and cell.decoding_seed.eq(row.decoding_seed).all()
        else:
            def progress(done, total):
                write_json(run / 'generation_progress.json', {
                    'temperature': row.temperature, 'top_p': row.top_p, 'done': done,
                    'total': total, 'cells_completed': len(frames), 'updated_utc': now()})
            strings = detokenize_generated(generate_smiles_batched(
                sampler_model, itos, stoi, cfg, gen_n=1000, temperature=row.temperature,
                top_p=row.top_p, seed=int(row.decoding_seed), batch_size=256, progress=progress), cfg)
            records = []
            for s in strings:
                mol = Chem.MolFromSmiles(s)
                canonical = None
                if mol is not None:
                    try:
                        canonical = Chem.MolToSmiles(mol)
                        if Chem.MolFromSmiles(canonical) is None:
                            canonical = None
                    except Exception:
                        canonical = None
                records.append({'raw_smiles': s, 'rdkit_valid': canonical is not None,
                                'canonical_smiles': canonical})
            cell = pd.DataFrame(records)
            cell['temperature'] = row.temperature
            cell['top_p'] = row.top_p
            cell['decoding_seed'] = int(row.decoding_seed)
            tmp = file.with_name(file.name + '.tmp.gz')
            cell.to_csv(tmp, index=False)
            tmp.replace(file)
        frames.append(cell)
        print('Completed cell', job['run_id'], row.temperature, row.top_p, flush=True)
    out = pd.concat(frames, ignore_index=True)
    assert len(out) == 9000 and out.groupby(['temperature', 'top_p']).size().eq(1000).all()
    out.to_csv(run / 'generated_molecules.csv.gz', index=False)
    frozen_check()
    assert tm['checkpoint_sha256'] == digest(run / tm['checkpoint'])
    write_json(run / 'generation_complete.json', {
        'run_id': job['run_id'], 'backbone': job['backbone'], 'condition': job['condition'],
        'training_seed': job['training_seed'], 'selected_epoch': 10, 'total_attempts': 9000,
        'attempts_per_point': 1000, 'n_grid_points': 9,
        'checkpoint_sha256': tm['checkpoint_sha256'],
        'generated_sha256': digest(run / 'generated_molecules.csv.gz'),
        'protocol_sha256': digest(HERE / 'protocol.json'),
        'decoding_seed_plan_sha256': digest(HERE / 'decoding_seed_plan.csv'),
        'used_native_eager_fallback': not identical,
        'wall_seconds': time.monotonic() - started, 'completed_utc': now()})


if __name__ == '__main__':
    main()

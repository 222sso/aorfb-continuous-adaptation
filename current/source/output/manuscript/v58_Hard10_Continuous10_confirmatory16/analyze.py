"""Execute the fixed three-test primary analysis only after all 96 runs pass audit."""
import json
import numpy as np
import pandas as pd
from scipy import stats
from study import HERE, PROTOCOL, JOBS, audit_score, digest, frozen_check, now, run_path, write_json


def paired_result(d):
    d = np.asarray(d, dtype=float)
    assert len(d) == 16 and np.isfinite(d).all()
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(16)
    if sd == 0:
        if mean == 0:
            return dict(mean_difference_pp=0., sample_sd_difference_pp=0., t=0.,
                        p_unadjusted=1., p_bonferroni3=1., ci95_lower_pp=0., ci95_upper_pp=0.,
                        simultaneous_ci_lower_pp=0., simultaneous_ci_upper_pp=0.,
                        superiority=False, test_case='all_differences_exactly_zero')
        return dict(mean_difference_pp=mean, sample_sd_difference_pp=0., t=None,
                    p_unadjusted=None, p_bonferroni3=None, ci95_lower_pp=None, ci95_upper_pp=None,
                    simultaneous_ci_lower_pp=None, simultaneous_ci_upper_pp=None,
                    superiority=False, test_case='degenerate_nonzero_difference_requires_review')
    tvalue = mean / se
    p = float(2 * stats.t.sf(abs(tvalue), df=15))
    p_adjusted = min(3 * p, 1.)
    c95 = float(stats.t.ppf(.975, df=15)) * se
    simultaneous = float(stats.t.ppf(1 - .05 / (2 * 3), df=15)) * se
    return dict(mean_difference_pp=mean, sample_sd_difference_pp=sd, t=tvalue,
                p_unadjusted=p, p_bonferroni3=p_adjusted,
                ci95_lower_pp=mean-c95, ci95_upper_pp=mean+c95,
                simultaneous_ci_lower_pp=mean-simultaneous, simultaneous_ci_upper_pp=mean+simultaneous,
                superiority=bool(mean > 0 and p_adjusted < .05), test_case='prespecified_paired_t')


def main():
    frozen_check()
    assert len(JOBS) == 96 and len(PROTOCOL['training_seeds']) == 16
    missing = [j['run_id'] for j in JOBS if not (run_path(j) / 'scoring_complete.json').exists()]
    if missing:
        raise RuntimeError(f'Primary inference locked: {len(missing)} of96 runs not complete. No partial p values computed.')
    if (HERE / 'FINAL_ANALYSIS_COMPLETE.json').exists():
        previous = json.loads((HERE / 'FINAL_ANALYSIS_COMPLETE.json').read_text())
        for name, expected in previous['output_hashes'].items():
            assert digest(HERE / name) == expected
        print('Previously completed final analysis preserved; not recomputed.')
        return
    seed_rows, cell_rows, audit_rows = [], [], []
    metrics = PROTOCOL['evaluation']['metrics']
    primary = PROTOCOL['statistics']['primary_metric']
    seeds = PROTOCOL['training_seeds']
    for job in JOBS:
        tm, gm, sm = audit_score(job)
        run = run_path(job)
        data = json.loads((run / 'seed_metrics.json').read_text())
        assert data['run_id'] == job['run_id'] and data['training_seed'] == job['training_seed']
        assert data['backbone'] == job['backbone'] and data['condition'] == job['condition']
        cells = pd.read_csv(run / 'cell_metrics_9.csv')
        assert len(cells) == 9 and cells.n_attempts.eq(1000).all()
        for m in metrics:
            assert np.isclose(data[m], cells[m].mean(), rtol=0, atol=1e-10, equal_nan=True)
        assert np.isfinite(data[primary])
        seed_rows.append(data)
        cell_rows.append(cells)
        audit_rows.append({
            'run_id': job['run_id'], 'backbone': job['backbone'], 'condition': job['condition'],
            'training_seed': job['training_seed'], 'selected_epoch': tm['selected_epoch'],
            'optimizer_updates': tm['optimizer_updates'], 'n_attempts': 9000,
            'pretrain_checkpoint_sha256': tm['pretrain_checkpoint_sha256'],
            'checkpoint_sha256': tm['checkpoint_sha256'], 'generated_sha256': gm['generated_sha256'],
            'scored_sha256': sm['scored_sha256'], 'training_jit_compile': tm['training_jit_compile'],
            'training_tf32_enabled': tm['training_tf32_enabled']})
    values = pd.DataFrame(seed_rows)
    cells = pd.concat(cell_rows, ignore_index=True)
    assert len(values) == 96 and len(cells) == 864 and cells.n_attempts.sum() == 864000
    assert values.groupby(['backbone', 'condition']).size().eq(16).all()
    endpoints = metrics + ['unique_predicted_targets_9000']
    summary = values.groupby(['backbone', 'condition'])[endpoints].agg(['mean', 'std']).reset_index()
    summary.columns = ['_'.join(c).rstrip('_') for c in summary.columns]
    pairs, tests, secondary = [], [], []
    for backbone in PROTOCOL['backbones']:
        block = values[values.backbone.eq(backbone)]
        c = block[block.condition.eq('continuous_targetmax10')].set_index('training_seed').loc[seeds]
        h = block[block.condition.eq('hard_fixed10')].set_index('training_seed').loc[seeds]
        for metric in endpoints:
            d = c[metric].to_numpy() - h[metric].to_numpy()
            for i, seed in enumerate(seeds):
                pairs.append({'backbone': backbone, 'metric': metric, 'training_seed': seed,
                              'Hard10': float(h[metric].iloc[i]), 'Continuous10': float(c[metric].iloc[i]),
                              'Continuous_minus_Hard': float(d[i])})
            if metric == primary:
                tests.append({'backbone': backbone, 'metric': metric, 'n_pairs': 16, 'df': 15,
                              'Hard10_mean_pct': float(h[metric].mean()),
                              'Continuous10_mean_pct': float(c[metric].mean()),
                              **paired_result(d)})
            else:
                secondary.append({'backbone': backbone, 'metric': metric, 'n_pairs': 16,
                                  'mean_difference': float(np.mean(d)),
                                  'sample_sd_difference': float(np.std(d, ddof=1)),
                                  'status': 'descriptive only, not a confirmatory test'})
    assert len(tests) == 3 and len(secondary) == 21
    result = pd.DataFrame(tests)
    values.to_csv(HERE / 'seed_metrics_96.csv', index=False)
    cells.to_csv(HERE / 'cell_metrics_864.csv', index=False)
    summary.to_csv(HERE / 'comparison_mean_sd.csv', index=False)
    pd.DataFrame(pairs).to_csv(HERE / 'paired_differences_all.csv', index=False)
    result.to_csv(HERE / 'primary_tests_3.csv', index=False)
    pd.DataFrame(secondary).to_csv(HERE / 'secondary_descriptive_21.csv', index=False)
    pd.DataFrame(audit_rows).to_csv(HERE / 'run_integrity_96.csv', index=False)
    degenerate = bool(result.test_case.eq('degenerate_nonzero_difference_requires_review').any())
    conclusion = {
        'status': 'requires_review_degenerate_primary' if degenerate else 'complete',
        'completed_utc': now(), 'new_training_runs': 96, 'new_seed_pairs': 48,
        'new_generated_attempts': 864000, 'prior_outcomes_included': False,
        'primary_tests': tests,
        'all_three_backbones_superior_on_primary': bool(result.superiority.all()) and not degenerate,
        'conditional_on_fixed_pretraining_and_frozen_screen': True,
        'no_DFT_or_manuscript_changes': True,
        'freeze_sha256': digest(HERE / 'EXECUTION_FROZEN.json'),
        'protocol_sha256': digest(HERE / 'protocol.json')}
    write_json(HERE / 'conclusion.json', conclusion)
    lines = ['Hard10–Continuous10 새 시드16쌍 독립 검증 결과', '',
             '총96회 학습,864000회 생성. 기존3시드 결과는 새 검정에 포함하지 않음.',
             '주평가: 신규 고유 예측 타깃 수율. 구조별 paired t, 양측, df15, Bonferroni3.',
             '동시 신뢰구간:98.333333%. 아래 차이는 Continuous10-Hard10, 단위%p.', '']
    for row in tests:
        lines.append(f"{row['backbone']}: Hard10 {row['Hard10_mean_pct']:.4f}%, Continuous10 {row['Continuous10_mean_pct']:.4f}%, 차이 {row['mean_difference_pp']:+.4f}%p")
        lines.append(f"  동시CI [{row['simultaneous_ci_lower_pp']}, {row['simultaneous_ci_upper_pp']}], 보정p={row['p_bonferroni3']}, 주평가 우위={row['superiority']}")
    lines += ['', '비타깃 자료 유지 자체의 인과효과, DFT물성, 전해질 실험성능에 대한 검증은 아님.',
              '1%p는 설계 참고폭. 통계적 유의성과 실질적 개선폭은 구분하여 해석할 것.',
              '나머지6지표와전체풀고유타깃수는기술통계.주검정을대체하지않음.']
    (HERE / 'Results_KO.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    frozen_check()
    outputs = ['seed_metrics_96.csv', 'cell_metrics_864.csv', 'comparison_mean_sd.csv',
               'paired_differences_all.csv', 'primary_tests_3.csv', 'secondary_descriptive_21.csv',
               'run_integrity_96.csv', 'conclusion.json', 'Results_KO.txt']
    write_json(HERE / 'FINAL_ANALYSIS_COMPLETE.json', {
        'status': conclusion['status'], 'completed_utc': now(),
        'n_runs': 96, 'n_pairs_per_backbone': 16, 'n_attempts': 864000,
        'output_hashes': {p: digest(HERE / p) for p in outputs}})
    print('All96runs audited; prespecified final analysis written.', flush=True)


if __name__ == '__main__':
    main()

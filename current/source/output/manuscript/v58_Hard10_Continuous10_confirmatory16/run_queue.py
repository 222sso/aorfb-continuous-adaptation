"""Durable local queue: the 96 authorized runs, followed by their frozen analysis."""
import concurrent.futures
import fcntl
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from study import HERE, ROOT, PIPE, PYTHON, JOBS, audit_generation, audit_score, audit_training, frozen_check, now, run_path, write_json

LOCK = threading.Lock()
STATUS = {j['run_id']: {'state': 'pending'} for j in JOBS}
ERRORS = []


def update(job, state, **extra):
    with LOCK:
        STATUS[job['run_id']] = {'state': state, 'updated_utc': now(), **extra}
        write_json(HERE / 'queue_status.json', {
            'updated_utc': now(), 'runner_pid': os.getpid(), 'total_jobs': 96,
            'jobs': STATUS})
    print(now(), job['run_id'], state, extra, flush=True)


def child(job, stage, cmd, env):
    run = run_path(job)
    with (run / f'{stage}.log').open('a') as log:
        log.write('\n' + now() + ' START ' + stage + '\n')
        log.flush()
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        update(job, stage, child_pid=p.pid)
        rc = p.wait()
        log.write('\n' + now() + f' EXIT {rc}\n')
        if rc:
            raise RuntimeError(f'{stage} failed with exit {rc}')


def run(job):
    env = os.environ.copy()
    env.update({'GPU_MEM_LIMIT_MB': '5000', 'OMP_NUM_THREADS': '2',
                'TF_NUM_INTRAOP_THREADS': '2', 'TF_NUM_INTEROP_THREADS': '2',
                'PYTHONHASHSEED': str(job['training_seed'])})
    failure_path = HERE / 'failures' / (job['run_id'] + '.json')
    failures = json.loads(failure_path.read_text()) if failure_path.exists() else []
    for attempt in range(len(failures), 2):
        try:
            frozen_check()
            path = run_path(job)
            if (path / 'scoring_complete.json').exists():
                audit_score(job)
                update(job, 'complete', attempts=9000, resumed_verified=True)
                return
            if not (path / 'training_complete.json').exists():
                if path.exists() and any(path.iterdir()):
                    # A failed/interrupted training never competes with a later run.
                    # Preserve it, then restart from the same original pretrained weights.
                    archived = HERE / 'failed_training_attempts' / job['run_id'] / str(time.time_ns())
                    archived.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(path), str(archived))
                path.mkdir(parents=True, exist_ok=True)
                cmd = [PYTHON, '-u', str(HERE / ('train_hard_fixed10.py' if job['arm'] == 'hard' else 'train_continuous_fixed10.py')),
                       '--dataset', 'reddb', '--arm', job['arm'], '--seed', str(job['training_seed']),
                       '--backbone', job['model'], '--tag', job['tag'],
                       '--p3a-config', str(PIPE / 'p3a_config.json'),
                       '--p3a-condition', job['p3a_condition'], '--p3a-operating-point', 'baseline']
                child(job, 'training', cmd, env)
            audit_training(job)
            if not (path / 'generation_complete.json').exists():
                child(job, 'generating', [PYTHON, '-u', str(HERE / 'generate.py'), '--run-id', job['run_id']], env)
            audit_generation(job)
            if not (path / 'scoring_complete.json').exists():
                child(job, 'scoring', [PYTHON, '-u', str(HERE / 'score_run.py'), '--run-id', job['run_id']], env)
            audit_score(job)
            update(job, 'complete', attempts=9000)
            return
        except Exception as exc:
            failure_path.parent.mkdir(exist_ok=True)
            failures.append({'recorded_utc': now(), 'error': repr(exc), 'attempt': attempt + 1})
            write_json(failure_path, failures)
            if attempt == 0:
                update(job, 'retry_same_seed_and_settings', error=repr(exc))
            else:
                update(job, 'failed', error=repr(exc))
                with LOCK:
                    ERRORS.append({'run_id': job['run_id'], 'error': repr(exc)})
                return
    update(job, 'failed', error='Two recorded technical failures; automatic retries exhausted.')
    with LOCK:
        ERRORS.append({'run_id': job['run_id'], 'error': 'retry budget exhausted'})


def main():
    lockfile = (HERE / 'queue.lock').open('w')
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen_check()
    assert len(JOBS) == 96
    write_json(HERE / 'RUNNING.json', {
        'status': 'running', 'pid': os.getpid(), 'started_utc': now(),
        'total_training_runs': 96, 'new_pairs_per_backbone': 16,
        'planned_generated_attempts': 864000})
    recurrent = queue.Queue()
    for job in JOBS:
        if job['backbone'] != 'TCN':
            recurrent.put(job)

    def recurrent_lane():
        while True:
            try:
                job = recurrent.get_nowait()
            except queue.Empty:
                return
            try:
                run(job)
            finally:
                recurrent.task_done()

    def tcn_lane():
        for job in JOBS:
            if job['backbone'] == 'TCN':
                run(job)
        recurrent_lane()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(recurrent_lane) for _ in range(3)] + [ex.submit(tcn_lane)]
            for f in concurrent.futures.as_completed(futures):
                f.result()
        if ERRORS:
            write_json(HERE / 'RUNNING.json', {'status': 'incomplete_technical_failures',
                       'finished_utc': now(), 'failures': ERRORS, 'final_inference_performed': False})
            raise RuntimeError(f'{len(ERRORS)} jobs failed; no final confirmatory inference.')
        frozen_check()
        with (HERE / 'final_analysis.log').open('a') as log:
            result = subprocess.run([PYTHON, '-u', str(HERE / 'analyze.py')], cwd=ROOT,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'OMP_NUM_THREADS': '2'})
        if result.returncode:
            raise RuntimeError('Final analysis failed; inspect final_analysis.log. No success declared.')
        final = json.loads((HERE / 'FINAL_ANALYSIS_COMPLETE.json').read_text())
        write_json(HERE / 'RUNNING.json', {'status': final['status'], 'finished_utc': now(),
                   'training_runs_completed': 96, 'generated_attempts': 864000})
        print('Requested study and fixed final analysis finished.', flush=True)
    except BaseException as exc:
        write_json(HERE / 'runner_exception.json', {'recorded_utc': now(), 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()

"""Read operational progress only, without computing interim treatment effects or p values."""
from collections import Counter
import json
import os
from study import HERE, JOBS, run_path


def main():
    queue_file = HERE / 'queue_status.json'
    data = json.loads(queue_file.read_text()) if queue_file.exists() else {}
    states = data.get('jobs', {})
    counts = Counter(states.get(j['run_id'], {}).get('state', 'pending') for j in JOBS)
    running_file = HERE / 'RUNNING.json'
    running = json.loads(running_file.read_text()) if running_file.exists() else {'status': 'not_started'}
    active = []
    for job in JOBS:
        state = states.get(job['run_id'], {})
        if state.get('state') not in ('training', 'generating', 'scoring'):
            continue
        path = run_path(job)
        progress_file = path / ('progress.json' if state['state'] == 'training' else 'generation_progress.json')
        try:
            progress = json.loads(progress_file.read_text()) if progress_file.exists() else {}
        except json.JSONDecodeError:
            progress = {'note': 'progress file being written'}
        pid = state.get('child_pid')
        alive = False
        if pid:
            try:
                os.kill(pid, 0)
                alive = True
            except ProcessLookupError:
                pass
        active.append({'run_id': job['run_id'], 'state': state['state'], 'child_pid': pid,
                       'process_alive': alive, 'progress': progress})
    output = {'study': running, 'counts': dict(counts), 'active': active,
              'verified_complete_runs': sum((run_path(j) / 'scoring_complete.json').exists() for j in JOBS),
              'final_analysis_complete': (HERE / 'FINAL_ANALYSIS_COMPLETE.json').exists()}
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

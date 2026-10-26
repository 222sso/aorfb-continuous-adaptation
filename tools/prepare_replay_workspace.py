"""Prepare an isolated copy of current inputs/code and print a command; never execute it."""
from pathlib import Path
import argparse, json, shutil, sys

ROOT=Path(__file__).resolve().parents[1]
REL=Path('output/manuscript/v58_Hard10_Continuous10_confirmatory16')
PIPE=Path('q1_comparison_experiments_260806/generality_replication_260816')
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--show-only',action='store_true',help='Print paths/commands without copying')
    a=p.parse_args();dest=a.destination.resolve()
    assert dest!=ROOT and ROOT not in dest.parents,'Choose a directory outside the release'
    jobs=json.loads((ROOT/'current/source'/REL/'jobs_96.json').read_text())
    job=next((j for j in jobs if j['run_id']==a.run_id),None)
    assert job is not None,'Run ID must be one of the96 recorded runs'
    if not a.show_only:
        assert not dest.exists(),'Destination must not already exist'
        shutil.copytree(ROOT/'current/source',dest,ignore=shutil.ignore_patterns('runs','__pycache__','*.pyc'))
    arm=job['arm'];script='train_hard_fixed10.py' if arm=='hard' else 'train_continuous_fixed10.py'
    command=[sys.executable,'-u',str(dest/REL/script),'--dataset','reddb','--arm',arm,'--seed',str(job['training_seed']),
             '--backbone',job['model'],'--tag',job['tag'],'--p3a-config',str(dest/PIPE/'p3a_config.json'),
             '--p3a-condition',job['p3a_condition'],'--p3a-operating-point','baseline']
    print(json.dumps({'workspace_prepared':not a.show_only,'working_directory':str(dest),
                      'environment':{'GPU_MEM_LIMIT_MB':'5000','OMP_NUM_THREADS':'2',
                                     'TF_NUM_INTRAOP_THREADS':'2','TF_NUM_INTEROP_THREADS':'2',
                                     'PYTHONHASHSEED':str(job['training_seed'])},
                      'training_command_argv':command,
                      'generation_command_argv':[sys.executable,str(dest/REL/'generate.py'),'--run-id',job['run_id']],
                      'scoring_command_argv':[sys.executable,str(dest/REL/'score_run.py'),'--run-id',job['run_id']],
                      'commands_executed':False},indent=2))
if __name__=='__main__':main()

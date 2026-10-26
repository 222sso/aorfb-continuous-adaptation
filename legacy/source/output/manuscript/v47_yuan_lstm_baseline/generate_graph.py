"""Run the unchanged sampler with an equivalent ordinary TensorFlow call graph.

No XLA, new sampler, incremental-state decoding, altered probability rule or RNG.
Remapping/arithmetic rewrites are disabled; all validation-position probabilities
matched the eager call bit-for-bit in generation_execution_audit.json.
"""
from pathlib import Path
import hashlib,json,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_case
tf=run_case.tf
tf.config.optimizer.set_experimental_options({'remapping':False,'arithmetic_optimization':False})
original_sampler=run_case.generate_smiles_batched
graphs={}
def same_sampler(model,*args,**kwargs):
    key=id(model)
    if key not in graphs:
        @tf.function(input_signature=[tf.TensorSpec((None,127),tf.int32)],jit_compile=False)
        def forward(inp):return model(inp,training=False)
        def call(inp,training=False):
            assert training is False
            return forward(tf.convert_to_tensor(inp,dtype=tf.int32))
        # Recheck the selected weights, not only the pretrained diagnostic model.
        # Any nonidentical execution falls back to the original eager call before
        # a single attempt is sampled. Correctness never depends on target scores.
        from ablation.vocab import encode_smiles_list
        vocab=json.loads((run_case.PIPE/'artifacts/vocab.json').read_text())
        corpus=run_case.pd.read_csv(run_case.PIPE/'artifacts/reddb.csv.gz')
        arr=encode_smiles_list(corpus.loc[corpus.split.eq('valid'),'encoded_smiles'].astype(str).tolist(),vocab['stoi'],run_case.CFG(),128)[:,:-1]
        exactly_equal=True;maximum=0.
        for start in range(0,len(arr),256):
            eager=model(arr[start:start+256],training=False).numpy()
            graph=call(arr[start:start+256]).numpy()
            exactly_equal &= bool(run_case.np.array_equal(eager,graph))
            maximum=max(maximum,float(run_case.np.max(run_case.np.abs(eager-graph))))
        graphs[key]=call if exactly_equal else model
        seed=int(sys.argv[sys.argv.index('--seed')+1])
        run_case.write_json(HERE/'runs'/f'seed_{seed}'/'selected_checkpoint_generation_parity.json',
            {'n_common_validation_inputs':len(arr),'all_position_probabilities_bitwise_equal':exactly_equal,
             'maximum_absolute_difference':maximum,'used_native_eager_fallback':not exactly_equal,
             'criterion':'Use the graph only for bitwise-identical outputs; otherwise use original eager call.',
             'no_fitting_or_new_molecule_sampling_in_check':True})
        print('Selected-checkpoint execution check:',seed,'bitwise_equal',exactly_equal,flush=True)
    return original_sampler(graphs[key],*args,**kwargs)
run_case.generate_smiles_batched=same_sampler
if __name__=='__main__':
    assert '--stage' in sys.argv and sys.argv[sys.argv.index('--stage')+1]=='generate'
    run_case.main()
    seed=int(sys.argv[sys.argv.index('--seed')+1]);path=HERE/'runs'/f'seed_{seed}'/'generation_complete.json'
    meta=json.loads(path.read_text())
    meta['execution_wrapper']='ordinary TensorFlow graph, XLA disabled; unchanged sampler and original default precision'
    meta['execution_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    meta['generation_execution_audit_sha256']=hashlib.sha256((HERE/'generation_execution_audit.json').read_bytes()).hexdigest()
    parity=json.loads((path.parent/'selected_checkpoint_generation_parity.json').read_text())
    meta['used_native_eager_fallback']=parity['used_native_eager_fallback']
    run_case.write_json(path,meta)

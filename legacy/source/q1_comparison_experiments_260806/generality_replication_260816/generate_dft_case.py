from __future__ import annotations

"""Generate for a DFT-target variant and save molecules; target judgement is deferred.

DFT endpoints cannot be computed for a generated molecule, so membership is decided
later by a predictor (see dft_decompose.py), exactly as in the RedDB pipeline. This
script only produces the molecules and their scaffold labels.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import tensorflow as tf
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ROOT = HERE.parents[1]
ARTIFACTS = HERE / "artifacts"
RESULTS = HERE / "results"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PARENT))

from ablation.config import CFG  # noqa: E402
from ablation.sample import detokenize_generated  # noqa: E402
from batch_sampling_v2 import generate_smiles_batched  # noqa: E402
import backbones  # noqa: E402

RDLogger.DisableLog("rdApp.*")

TEMPERATURES = (0.70, 0.90, 1.10)
TOP_PS = (0.75, 0.85, 0.95)
GEN_N = 1000


def configure_gpu() -> None:
    """Stop TensorFlow preallocating the whole device.

    Without this a sampling process reserves the entire GPU even though the model is
    tiny. GPU_MEM_LIMIT_MB additionally imposes a hard ceiling.
    """
    import os
    limit = os.environ.get("GPU_MEM_LIMIT_MB")
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            if limit:
                tf.config.set_logical_device_configuration(
                    gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=int(limit))]
                )
            else:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def main() -> None:
    configure_gpu()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--arm", required=True, choices=("equal", "soft", "hard"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tag", default=None, help="output subdirectory suffix")
    parser.add_argument("--vocab", default="vocab.json")
    backbones.add_argument(parser)
    args = parser.parse_args()

    stem = args.dataset if not args.tag else f"{args.dataset}_{args.tag}"
    run_dir = RESULTS / stem / args.arm / f"seed_{args.seed}"
    training_complete = run_dir / "training_complete.json"
    if not training_complete.exists():
        raise FileNotFoundError(f"Train first: {training_complete}")
    complete = run_dir / "generation_complete.json"
    if complete.exists():
        print(f"SKIP {args.dataset}/{args.arm}/seed{args.seed}")
        return

    cfg = CFG()
    vocab = json.loads((ARTIFACTS / args.vocab).read_text(encoding="utf-8"))
    stoi = {str(k): int(v) for k, v in vocab["stoi"].items()}
    itos = {int(k): str(v) for k, v in vocab["itos"].items()}

    corpus = pd.read_csv(ARTIFACTS / f"{args.dataset}.csv.gz")
    # fillna(""): an acyclic molecule has an empty Murcko scaffold, which pandas
    # reads back from the artifact as NaN. astype(str) alone turns those into the
    # literal "nan", so a generated acyclic molecule (scaffold "") never matches the
    # corpus's own acyclic members and is wrongly counted as a new scaffold.
    corpus_scaffolds = set(corpus.scaffold.fillna("").astype(str))
    train_smiles = set(corpus.loc[corpus.split == "train", "canonical_smiles"].astype(str))

    training_meta = json.loads(training_complete.read_text())
    trained_backbone = training_meta.get("backbone", "A_TCN")
    if trained_backbone != args.backbone:
        raise ValueError(
            f"Backbone mismatch: checkpoint is {trained_backbone}, requested {args.backbone}"
        )
    model = backbones.build(args.backbone, len(stoi), cfg)
    model.load_weights(run_dir / training_meta["checkpoint"])

    frames = []
    for temperature in TEMPERATURES:
        for top_p in TOP_PS:
            gen_seed = abs(hash((args.dataset, args.arm, args.seed, temperature, top_p))) % (2**31)
            if args.tag and args.tag.startswith("p3a_"):
                key = f"{args.dataset}|{args.arm}|{args.seed}|{temperature}|{top_p}|{args.tag}"
                gen_seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (2**31)
            plain = detokenize_generated(
                generate_smiles_batched(model, itos, stoi, cfg, gen_n=GEN_N,
                                        temperature=temperature, top_p=top_p,
                                        seed=gen_seed, batch_size=256), cfg)
            rows = []
            for smiles in plain:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    rows.append({"rdkit_valid": False, "canonical_smiles": None, "scaffold": None})
                    continue
                try:
                    rows.append({"rdkit_valid": True,
                                 "canonical_smiles": Chem.MolToSmiles(mol),
                                 "scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=mol)})
                except Exception:
                    rows.append({"rdkit_valid": False, "canonical_smiles": None, "scaffold": None})
            frame = pd.DataFrame(rows)
            frame["temperature"] = temperature
            frame["top_p"] = top_p
            frame["new_scaffold"] = (
                frame.rdkit_valid & ~frame.scaffold.fillna("").isin(corpus_scaffolds)
            )
            frame["novel_molecule"] = frame.rdkit_valid & ~frame.canonical_smiles.isin(train_smiles)
            frames.append(frame)
            print(f"[{args.dataset}/{args.arm}/s{args.seed}] T={temperature} p={top_p} "
                  f"valid={100*frame.rdkit_valid.mean():.1f}% "
                  f"newScaffold={100*frame.new_scaffold.mean():.1f}%")

    molecules = pd.concat(frames, ignore_index=True)
    molecules.to_csv(run_dir / "generated_molecules.csv.gz", index=False)
    complete.write_text(json.dumps({
        "dataset": args.dataset, "arm": args.arm, "training_seed": args.seed,
        "backbone": args.backbone,
        "n_grid_points": len(TEMPERATURES) * len(TOP_PS), "attempts_per_point": GEN_N,
        "total_attempts": GEN_N * len(TEMPERATURES) * len(TOP_PS),
        "target_judgement": "deferred to a DFT property predictor (dft_decompose.py)",
        "validity_pct": float(100 * molecules.rdkit_valid.mean()),
    }, indent=2), encoding="utf-8")
    print(f"[done] {run_dir}")


if __name__ == "__main__":
    main()

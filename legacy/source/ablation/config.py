# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class CFG:
    CHEMBL_CSV: str = "inputs/sampled_60000_chembl_smiles.csv"
    REDDB_CSV:  str = "dist_reports/merged_all_inchikey_v2.csv"
    REDDB_SMILES_COL: str = "smiles_main"
    REDDB_ENERGY_COL: str = "Solvation_energy"

    OUTDIR: str = "./tcnlstm_transfer_outputs_260311"

    VALID_RATIO_CHEMBL: float = 0.005
    VALID_RATIO_REDDB:  float = 0.01

    REDDB_FRAC: float = 1.0
    REDDB_SELECT_LOWEST: bool = False  # True=가장 낮은(더 negative) 에너지 10%

    MIN_LEN: int = 5
    MAX_LEN: int = 128
    DROP_TOO_LONG: bool = True

    PAD: str = "~"
    BOS: str = "^"
    EOS: str = "$"
    UNK: str = "?"

    TOK_SO3H: str = "¤"
    TOK_COOH: str = "§"

    SEED: int = 42
    BATCH_SIZE: int = 128
    BATCH_SIZE_FINETUNE: int = 32
    PRETRAIN_EPOCHS: int = 100
    FINETUNE_EPOCHS: int = 10
    LR_PRETRAIN: float = 5e-4
    LR_FINETUNE: float = 1e-4

    GEN_TOP_P: float = 0.95
    GEN_TEMPERATURE: float = 1.20
    GEN_N: int = 500
    # Sampling sweep for generation-robustness study (fixed ckpt, varied decoding)
    GEN_SWEEP_ENABLE: bool = True
    GEN_SWEEP_N: int = 1000
    GEN_SWEEP_TEMPERATURES: Tuple[float, ...] = (0.70, 0.90, 1.10)
    GEN_SWEEP_TOP_PS: Tuple[float, ...] = (0.75, 0.85, 0.95)

    FINETUNE_MIX_RATIO: float = 0.5   # RedDB:ChEMBL = 7:3
    FINETUNE_MIX_STEPS: int = 800     # 1 epoch당 step 수(고정)

    REDDB_W_THR: float = 1.2
    REDDB_P_HI: float = 0.90
    
    REDDB_FRAC_BOOST: float = 1.0   # 1.0이면 boost 안 함
    REDDB_USE_FRAC_FILTER: bool = False  # False면 select_fraction_by_energy 안 함

    # ============================================================
    # 0) CONFIG extras (softft weights)
    # ============================================================
    WEIGHT_CSV: str = "dist_reports/softft_v3/softft_scores_weights.csv"
    W_COL: str = "w"
    W_USE: str = "use_for_ft"
    USE_ONLY_SOFTFT: bool = False  # True면 w_map에 존재하는 inchiKey만 RedDB finetune에 사용

    # ============================================================
    # 4) (Optional) Augmentation params
    # ============================================================
    CHEMBL_AUG_N: int = 1
    CHEMBL_AUG_PROB: float = 0.5
    REDDB_AUG_N: int = 1
    REDDB_AUG_PROB: float = 0.1
    
    PRED_DIR: str = "./pred_outputs_multitask_scaffoldsplit_repeat"
    PRED_SEEDS: List[int] = (42, 43, 44)   # 또는 [42,43,44]
    NOVEL_REF_PATH: str = "dist_reports/merged_all_inchikey_v2.csv"
    
    USE_FT_WEIGHTS: bool = True   # baseline=True, no-weight=False
    EXP_TAG: str = "exp_baseline" # 출력 폴더 구분용

def ensure_outdir(cfg: CFG):
    os.makedirs(cfg.OUTDIR, exist_ok=True)

GEN_GRID = {
    "gen_n": [500],
    "temperature": [1.20],
    "top_p": [0.95],
}

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _draw_top_p(probabilities: np.ndarray, top_p: float, temperature: float, rng: np.random.Generator) -> int:
    probs = np.asarray(probabilities, dtype=np.float64)
    logits = np.log(np.maximum(probs, 1e-12)) / max(float(temperature), 1e-8)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    order = np.argsort(probs)[::-1]
    cutoff = int(np.searchsorted(np.cumsum(probs[order]), float(top_p))) + 1
    keep = order[: max(1, min(cutoff, len(order)))]
    kept = probs[keep]
    kept /= kept.sum()
    return int(rng.choice(keep, p=kept))


def generate_smiles_batched(
    model,
    itos: dict[int, str],
    stoi: dict[str, int],
    cfg,
    gen_n: int,
    temperature: float,
    top_p: float,
    seed: int,
    batch_size: int = 256,
    progress: Callable[[int, int], None] | None = None,
) -> list[str]:
    """Batch-equivalent autoregressive sampling for all new comparison arms."""

    total = int(gen_n)
    max_input = int(cfg.MAX_LEN) - 1
    pad_id = int(stoi[cfg.PAD])
    bos_id = int(stoi[cfg.BOS])
    eos_id = int(stoi[cfg.EOS])
    rng = np.random.default_rng(int(seed))
    outputs: list[str] = []

    for start in range(0, total, int(batch_size)):
        current = min(int(batch_size), total - start)
        x = np.full((current, max_input), pad_id, dtype=np.int32)
        x[:, 0] = bos_id
        active = np.ones(current, dtype=bool)
        generated: list[list[int]] = [[] for _ in range(current)]

        for step in range(max_input):
            if not active.any():
                break
            probabilities = model(x, training=False).numpy()[:, step, :]
            for row in np.flatnonzero(active):
                token = _draw_top_p(probabilities[row], top_p, temperature, rng)
                if token == eos_id:
                    active[row] = False
                else:
                    generated[row].append(token)
                    if step + 1 < max_input:
                        x[row, step + 1] = token

        for ids in generated:
            outputs.append("".join(itos[i] for i in ids if i not in (pad_id, bos_id, eos_id)))
        if progress:
            progress(min(start + current, total), total)

    return outputs

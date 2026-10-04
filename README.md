# JEPA Interpretability Experiment

A scaled-down Joint-Embedding Predictive Architecture (JEPA) trained on MNIST with forward-hook causal interventions across top-k SVD directions of the latent space.

**Protocol**: Frozen splits · 5 seeds (42, 100, 2026, 3141, 404) · 2,000-sample test set · Mean ± SD reporting. Full spec: [PROTOCOL.md](PROTOCOL.md).

---

## Architecture

```
Context (left 392px) → Context Encoder (MLP) → Latent z (dim=64) → Predictor (MLP) → Predicted Latent
                                                      ↑
                                            [Hook Interventions]
Target  (right 392px) → Target Encoder (EMA) ──────────────────────────────────────→ Target Latent
```

**Objective**: Predict the target encoder representation from context latent via MSE.  
**SVD**: Mean-centered SVD on test latents extracts orthonormal directions v₁, …, vₖ.

**Interventions** (all benchmarked against a matched random-unit-vector control):

| Type | Operation |
|---|---|
| Ablation (Projection-Out) | z − (z·v)v |
| Clamping (Mean-Projection) | z + (μ_{z·v} − z·v)v |
| Amplification (2×) | z + v(z·v) |
| Swapping | shuffle magnitudes across batch |

**Causal Delta** = ΔLoss_feature − ΔLoss_control

---

## Quick Start

```bash
git clone https://github.com/NeuralAlchemist-ai/JEPA-interpretability-experiment.git
cd JEPA-interpretability-experiment
uv sync
uv run python run.py                              # full 5-seed run
uv run python run.py --seeds 42 --epochs 5 --top-k 3  # fast single-seed
```

### CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--seeds` | `42 100 2026 3141 404` | Random seeds for multi-seed averaging |
| `--epochs` | `5` | Training epochs per seed |
| `--batch-size` | `128` | Train/test batch size |
| `--top-k` | `5` | Number of top SVD directions |
| `--output-dir` | `outputs` | Output directory |
| `--lr` | `1e-3` | Learning rate (AdamW) |
| `--latent-dim` | `64` | Latent bottleneck dimension |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--skip-plots` | `False` | Skip plot rendering |

---

## Frozen Protocol

- **Train split**: 10,000 samples · `data/splits/train_indices.json` (seed 42)
- **Test split**: 2,000 samples · `data/splits/test_indices.json` (seed 42)
- **Zero leakage**: train/test index pools are strictly disjoint

---

## Outputs

```
outputs/
├── experiment.log                   # Timestamped execution log
├── frozen_protocol.json             # Protocol manifest + dataset hashes
├── intervention_table.csv           # Rank-1 results (mean + std columns)
├── intervention_table_summary.csv   # Full multi-rank summary (Mean ± SD strings)
├── intervention_table_raw.csv       # Granular per-seed/per-rank data
├── svd_variance_spectrum.csv        # Explained variance % per SVD rank
└── plots/
    ├── 1_variance_vs_causal_delta.png      # Explained variance vs Causal Delta scatter
    ├── 2_causal_delta_across_ranks.png     # Causal effect decay across SVD ranks
    ├── 3_seed_stability.png                # Rank-1 ablation stability across seeds
    ├── 4_svd_vs_random_control.png         # SVD direction vs matched random control (dumbbell)
    └── 5_causal_effect_distribution.png    # Seed-level distribution of causal effects
```

See [RESULT.md](RESULT.md) for findings and analysis.

---

## License

MIT

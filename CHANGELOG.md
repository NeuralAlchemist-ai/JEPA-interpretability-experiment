## Protocol Update: Publication Figures Mapping

> **Note on Protocol Deviation:** The visualization pipeline was updated to better capture seed-level stability, SVD-specific controls, and distribution traits discovered during implementation.

> **Note on Protocol Execution:** The experiment can be run using `python run.py`.

### Figure Mapping & Justification

| Frozen Protocol Target (`outputs/plots/`) | Current Implementation (`outputs/plots/`) | Conceptual Shift / Scientific Justification |
| :--- | :--- | :--- |
| `1_effect_size_comparison.png` - Target Feature Effect vs. Matched Control across interventions | `1_variance_vs_causal_delta.png` | Replaced the intervention-level effect-size comparison with a combined analysis of representation variance and causal effect. This directly addresses whether high-variance latent directions are also causally important. |
| `2_causal_delta.png` - Net Causal Delta with relative MSE degradation | `2_causal_delta_across_ranks.png` | Shifted emphasis from a single Rank-1 causal effect to the causal-effect profile across explicit SVD ranks, making the concentration and decay of causal necessity visible. |
| `3_variance_vs_causal_necessity.png` - Dual-panel representation variance versus causal necessity | `3_seed_stability.png` | Replaced the dual-panel summary with a seed-level stability analysis of Rank-1 ablation. This makes the reproducibility of the observed causal effect across independent initializations more explicit. |
| `4_causal_decay_spectrum.png` - Multi-rank causal decay curves across interventions | `4_svd_vs_random_control.png` | Reframed the multi-rank analysis around the central causal-control comparison: whether SVD directions produce effects beyond matched random directions. |
| `5_mechanism_disentanglement.png` - Global scaling vs. instance-level variation | `5_causal_effect_distribution.png` | Replaced the mechanism-disentanglement visualization with a distribution of seed-level causal effects. This emphasizes robustness across experimental initializations rather than introducing a separate mechanistic interpretation. |

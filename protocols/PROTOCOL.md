# JEPA Causal Interpretability Evaluation Protocol v1

## Status

**Protocol status:** Frozen specification
**Experiment status:** In progress

This protocol defines the bounded JEPA causal interpretability evaluation on MNIST. It establishes the pre-registered methodology for isolating whether learned latent representation directions exert genuine causal control over downstream predictions or merely reflect non-causal statistical correlations.

The experimental specification, model architectures, dataset splits, interventions, metrics, and decision rules are frozen.

## 1. Research Question

Do principal latent representation directions in a Joint-Embedding Predictive Architecture (JEPA) causally drive downstream prediction, or do they represent non-causal correlational artifacts?

The evaluation tests computational necessity and dynamic-range sensitivity by applying targeted geometric interventions to latent representations at the context-predictor bottleneck and evaluating against matched-random unit vector controls.

## 2. Failure Mode

**Failure mode:** Conflation of representation variance with causal necessity.

In unsupervised self-supervised learning, high-variance directions identified by spectral decomposition (e.g., SVD/PCA) are frequently assumed to be the most functionally important. However, variance across latent dimensions does not necessarily imply causal utility for downstream computation.

The failure mode under test is the presence of non-causal representation directions: latent axes that explain significant data variance but whose perturbation does not affect downstream predictor performance beyond a random isotropic perturbation.

Arbitrary unnormalized perturbations, unconstrained adversarial noise, and interventions lacking matched-random controls are excluded from this protocol.

## 3. Models

The evaluation targets a scaled-down Joint-Embedding Predictive Architecture (JEPA) trained on split-image prediction:

* **Task**: Predicting the latent representation of the right image half (target) from the left image half (context).
* **Context Encoder**: Linear(392 → 196) → ReLU → Linear(196 → 64).
* **Target Encoder**: Exact duplicate architecture initialized with context weights; updated purely via Exponential Moving Average (EMA) with $\beta = 0.99$ (no backpropagation gradients).
* **Predictor**: Linear(64 → 64) → ReLU → Linear(64 → 64).
* **Latent Bottleneck Dimension**: $D = 64$.

Model parameters, layer widths, and activations are fixed.

No architecture adjustments, hyperparameter search, or post-hoc model modifications are introduced.

## 4. Dataset and Fixed Splits

Dataset:

* MNIST (`torchvision.datasets.MNIST`)
* 10 classes
* Image dimensions: $28 \times 28$ grayscale, normalized to $[-1, 1]$
* Spatial partitioning:
  * Context half: left $28 \times 14 = 392$ pixels
  * Target half: right $28 \times 14 = 392$ pixels

Fixed dataset budget:

* Training examples: **10,000**
* Test examples: **2,000**

The committed manifests are:

```text
data/splits/train_indices.json
data/splits/test_indices.json
```

They contain the first 10,000 MNIST training indices (SHA-256 short: `7d99a727ec84`) and first 2,000 MNIST test indices (SHA-256 short: `edd6c0024ff4`). This is a deterministic fixed subset, not a newly randomized split.

Zero data leakage is strictly enforced:

$$\text{train\_indices} \cap \text{test\_indices} = \emptyset$$

All protocol seeds train on exactly the same 10,000 training indices.

All model evaluations span the entire fixed test set of 2,000 examples (16 batches of batch size 128).

No examples may be added, removed, or replaced after the protocol is frozen.

## 5. Representation Analysis and Geometric Interventions

### SVD Feature Extraction

Prior to intervention, test context representations across all 2,000 test examples are accumulated and mean-centered:

$$\tilde{Z} = Z - \bar{Z} \in \mathbb{R}^{N \times D}$$

Singular Value Decomposition is computed:

$$\tilde{Z} = U \Sigma V^T$$

The rows of $V^T$ define the orthonormal principal directions $v_1, v_2, \dots, v_k \in \mathbb{R}^D$, ranked by explained variance ratio:

$$\lambda_r = \frac{\sigma_r^2}{\sum_{j=1}^D \sigma_j^2}$$

Evaluation is bounded to the top-5 principal directions ($k = 5$, Ranks 1 through 5).

### Targeted Interventions

For each principal direction $v_r$, four deterministic geometric interventions are applied to bottleneck representations $z \in \mathbb{R}^D$:

1. **Ablation (Projection-Out)**:
   $$z_{\text{ablated}} = z - (z \cdot v) v$$
   *Tests complete computational necessity of the axis.*

2. **Amplification (2×)**:
   $$z_{\text{amplified}} = z + v (z \cdot v)$$
   *Tests dynamic range calibration and sensitivity along the axis.*

3. **Clamping (Mean-Projection)**:
   $$z_{\text{clamped}} = z + (\mu_{z \cdot v} - z \cdot v) v$$
   *Tests necessity of per-sample instance variance versus static baseline coordinate presence.*

4. **Swapping**:
   $$z_{\text{swapped}} = z - (z \cdot v) v + (z_{\pi(i)} \cdot v) v$$
   *Shuffles feature coordinate magnitudes across batch instances using a deterministic permutation $\pi$.*

### Excluded Interventions

The following are excluded:

* Unbounded or arbitrary magnitude shifts
* Non-orthogonal directional transformations
* Dynamic interventions adapted during evaluation
* Probe retraining or predictor adaptation on intervened activations

## 6. Training Conditions

Training configuration for JEPA across all seeds:

| Setting | Value |
| --- | --- |
| Training samples | 10,000 |
| Batch size | 128 |
| Training epochs | 5 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 0.01 |
| Target EMA beta ($\beta$) | 0.99 |
| Loss function | Mean Squared Error in latent representation space |
| Loss reduction | Mean over latent dimensions and batch |
| Target encoder gradients | None (updated solely via EMA) |
| Optimizer scope | `context_encoder` and `predictor` parameters |
| Seeds | 42, 100, 2026, 3141, 404 |

Loss formulation:

$$\mathcal{L} = \frac{1}{D} \sum_{j=1}^D \left( \hat{z}_j^{\text{pred}} - z_j^{\text{target}} \right)^2$$

Target encoder parameter update:

$$\theta_{\text{target}} \leftarrow 0.99 \cdot \theta_{\text{target}} + 0.01 \cdot \theta_{\text{context}}$$

No hyperparameter tuning or early stopping is performed against test set representations.

## 7. Matched Random Control

To separate true axis-specific computational dependence from general loss sensitivity (e.g., loss of representation norm or arbitrary subspace disturbance), every intervention is evaluated against a matched-random control direction:

$$u_r \sim \mathcal{N}(0, I_D), \quad \|u_r\|_2 = 1$$

Control specification:

* Direction generation is deterministic: seeded with `seed + 1000 + rank`.
* The control unit vector $u_r$ undergoes the exact same mathematical transformation as $v_r$ (ablation, amplification, clamping, swapping).
* The control effect size serves as the empirical noise floor for that intervention.

The primary conclusion must evaluate the net effect over this matched control.

## 8. Causal Instrumentation and Hooking Protocol

Representations are intercepted live at the latent bottleneck between the context encoder and predictor:

* **Hook location**: `context_encoder[-1]` (output of the final Linear layer).
* **Mechanism**: PyTorch `register_forward_hook`.
* **Execution**: Forward hooks dynamically modify bottleneck activations during evaluation and are removed immediately after batch pass completion.
* **Evaluation scope**: Multi-batch pass across all 2,000 test set images (16 batches of 128).
* **Predictor state**: Frozen in evaluation mode (`model.eval()`). No gradients are computed, and predictor weights are never fine-tuned or adapted to intervened activations.

## 9. Primary Metric

The primary metric is **Causal Delta** ($\Delta_{\text{causal}}$), defined as the net prediction degradation directly attributable to the geometric feature axis, above the matched-random control:

For a given intervention operation and direction:

$$\Delta \mathcal{L}_{\text{feature}} = \mathcal{L}_{\text{intervened}} - \mathcal{L}_{\text{baseline}}$$

$$\Delta \mathcal{L}_{\text{control}} = \mathcal{L}_{\text{control}} - \mathcal{L}_{\text{baseline}}$$

$$\text{Causal Delta} = \Delta \mathcal{L}_{\text{feature}} - \Delta \mathcal{L}_{\text{control}}$$

Positive Causal Delta indicates that perturbing the specific SVD axis degrades downstream target prediction significantly more than perturbing an arbitrary random direction of identical norm.

For each seed and SVD rank, metrics are calculated across all 2,000 test examples.

The final analysis reports:

* Mean Causal Delta across the 5 canonical seeds
* Sample standard deviation across seeds
* Paired effect size comparison ($\Delta \mathcal{L}_{\text{feature}}$ vs. $\Delta \mathcal{L}_{\text{control}}$)

## 10. Frozen Decision Rule

Before inspecting intervention outcomes, the following classification rule is fixed:

An SVD rank direction is classified under one of three causal statuses:

1. **Strictly Causal**:
   * Mean Ablation Causal Delta satisfies:
     $$\text{mean}(\Delta_{\text{causal}}^{\text{ablation}}) > +0.50$$
   * Ablation Causal Delta exceeds the random control effect by at least $10\times$:
     $$\frac{\Delta \mathcal{L}_{\text{feature}}}{\Delta \mathcal{L}_{\text{control}}} \ge 10.0$$
   * Mean Amplification Causal Delta is positive:
     $$\text{mean}(\Delta_{\text{causal}}^{\text{amplification}}) > 0$$

2. **Weakly Causal**:
   * Mean Ablation Causal Delta satisfies:
     $$+0.05 \le \text{mean}(\Delta_{\text{causal}}^{\text{ablation}}) \le +0.50$$
   * Ablation Causal Delta remains reliably above the control noise floor across seeds.

3. **Non-Causal / Noise Floor (Merely Correlated)**:
   * Mean Ablation Causal Delta satisfies:
     $$\text{mean}(\Delta_{\text{causal}}^{\text{ablation}}) < +0.05$$
   * The 95% confidence interval or standard deviation includes zero or overlaps with the matched random control.

Directions classified as Non-Causal demonstrate that high representation variance does not imply downstream computational utility.

## 11. Secondary Metrics

The following metrics are recorded as secondary representation diagnostics:

* **SVD Explained Variance Ratio ($\lambda_r$)**: fraction of total representation variance explained by each rank.
* **Baseline Prediction Loss ($\mathcal{L}_{\text{baseline}}$)**: unperturbed test set MSE on target representations.
* **Relative Loss Degradation**:
  $$\text{Relative Degradation} = \frac{\Delta_{\text{causal}}}{\mathcal{L}_{\text{baseline}}} \times 100\%$$
* **Multi-Rank Causal Decay Spectrum**: progression of Causal Delta from Rank 1 to Rank 5 across all 4 interventions.
* **Mechanism Disentanglement Ratio**: comparison of global coordinate scaling (Ablation, Amplification) versus instance-level variation (Clamping, Swapping).

These secondary metrics characterize the nature of representation encoding but do not override the primary decision rule.

## 12. Seeds and Reproducibility

The exact seeds are:

```text
42
100
2026
3141
404
```

Seeds must be applied consistently to:

* Python `random` state
* NumPy `np.random` state
* PyTorch CPU random state
* CUDA random state where available
* Model weight initialization
* Training data-loader shuffling
* Matched-random control generation (`seed + 1000 + rank`)
* Swapping permutation generation (`seed + rank`)

Deterministic PyTorch settings must be maintained throughout training and evaluation.

Each seed execution contributes paired observations across all ranks and operations.

Existing result files must not be silently overwritten without logging.

## 13. Parameter and Compute Matching

Model parameter configurations:

| Component | Architecture | Parameter Count |
| --- | --- | --- |
| Context Encoder | Linear(392→196) + ReLU + Linear(196→64) | 89,636 |
| Target Encoder | Linear(392→196) + ReLU + Linear(196→64) (EMA) | 89,636 (non-trainable) |
| Predictor | Linear(64→64) + ReLU + Linear(64→64) | 8,320 |
| **Total Trainable** | Context Encoder + Predictor | **97,956** |

Compute budget:

* Training budget: exactly 5 epochs per seed (390 optimizer steps per seed at batch size 128)
* Evaluation budget: exactly 16 batches (2,000 samples) per intervention, rank, and seed

No architecture or compute-budget adjustment may be made after seeing results.

## 14. Artifacts to Retain

For every seed and aggregate run, retain structured artifacts:

1. **Manifest and Configuration**:
   * `outputs/frozen_protocol.json`: verification manifest containing split hashes, sample counts, and seeds.

2. **Numerical Tables**:
   * `outputs/intervention_table_raw.csv`: per-seed, per-rank, per-operation records (Base Loss, Feature Loss, Control Loss, Causal Delta).
   * `outputs/intervention_table_summary.csv`: aggregated mean ± standard deviation table across all 5 seeds.
   * `outputs/intervention_table.csv`: primary numeric table for SVD Rank-1 interventions.
   * `outputs/svd_variance_spectrum.csv`: explained variance mean and standard deviation across SVD ranks.

3. **Publication Figures** (`outputs/plots/`):
   * `1_variance_vs_causal_delta.png`: Combined representation variance and causal effect, showing how explained variance relates to causal necessity across SVD ranks.
   * `2_causal_delta_across_ranks.png`: Causal Delta across explicit SVD ranks, showing the decay of causal effects across the latent representation.
   * `3_seed_stability.png`: Seed-level stability of the Rank-1 ablation effect, showing whether the observed causal effect is consistent across random initializations.
   * `4_svd_vs_random_control.png`: Comparison of SVD-direction effects against matched random-control effects across ranks.
   * `5_causal_effect_distribution.png`: Distribution of seed-level causal effects, emphasizing robustness of the causal intervention results across experimental initializations.

4. **Execution Log**:
   * `outputs/experiment.log`: comprehensive timestamped execution log.

## 15. Pre-Outcome Freeze

Before result interpretation, the following manifests and specifications must be committed:

```text
PROTOCOL.md
frozen_protocol.json
data/splits/train_indices.json
data/splits/test_indices.json
```

along with all implementation code in `jepa/` and `interpretability/`.

No outcome-bearing conclusion may be drawn from uncommitted or modified protocol configurations.

Any change to split indices, model dimensions, intervention definitions, or decision thresholds requires a protocol version increment (`v2`) and re-execution.

## 16. Reproduction

The complete experiment is executable from a clean environment using one documented command:

```bash
python run.py
```

To explicitly specify the frozen parameters:

```bash
python run.py \
  --epochs 5 \
  --seeds 42 100 2026 3141 404 \
  --batch-size 128 \
  --top-k 5 \
  --lr 0.001 \
  --latent-dim 64 \
  --output-dir outputs
```

The reproduction command must produce identical numerical outputs within floating-point tolerance on deterministic hardware.

## 17. Interpretation Limits

This experiment evaluates targeted geometric interventions on a scaled-down JEPA trained on MNIST digit halves.

A result satisfying the decision rule supports the conclusion that downstream target prediction causally depends on a low-dimensional principal subspace (Rank 1), while other variance-explaining directions represent non-causal statistical correlations.

It does not by itself establish:

* Identical causal spectrum behavior across arbitrary image datasets or continuous domains.
* Invariance of causal hierarchies under alternative self-supervised objectives (e.g., contrastive learning, masked autoencoding).
* Behavior of full-scale vision architectures (e.g., ViT-based I-JEPA or V-JEPA) without explicit testing.
* Causal disentanglement along non-orthogonal or non-linear manifolds not captured by SVD.

Negative or null causal deltas must be reported as evidence of non-causal representation directions rather than discarded or tuned away.

## 18. Approval Gate

**Current state:** Verified / Frozen.

The protocol specification, model architecture, data splits, intervention definitions, matched controls, metrics, and decision rules are frozen. Any subsequent experiment must reference this document or formally branch to a new version.

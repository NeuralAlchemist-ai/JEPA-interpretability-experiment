#!/usr/bin/env python

import argparse
import logging
import os
import random
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

from interpretability.interventions import CausalIntervention
from interpretability.visualize import generate_plots
from jepa.data import (
    PROTOCOL_SEEDS,
    PROTOCOL_TEST_SAMPLES,
    PROTOCOL_TRAIN_SAMPLES,
    _load_split_indices,
    get_mnist_loaders,
    get_standard_test_loader,
    split_context_target,
)
from jepa.model import JEPA


def verify_frozen_protocol(data_root: str, output_dir: str, logger: logging.Logger) -> None:
    logger.info("============================================================")
    logger.info("VERIFYING FROZEN PROTOCOL")
    logger.info("============================================================")

    train_indices = _load_split_indices("train", root=data_root)
    test_indices = _load_split_indices("test", root=data_root)

    assert len(train_indices) == PROTOCOL_TRAIN_SAMPLES, (
        f"Train sample count mismatch: {len(train_indices)} vs expected {PROTOCOL_TRAIN_SAMPLES}"
    )
    assert len(test_indices) == PROTOCOL_TEST_SAMPLES, (
        f"Test sample count mismatch: {len(test_indices)} vs expected {PROTOCOL_TEST_SAMPLES}"
    )

    import hashlib
    train_hash = hashlib.sha256(json.dumps(train_indices).encode("utf-8")).hexdigest()[:12]
    test_hash = hashlib.sha256(json.dumps(test_indices).encode("utf-8")).hexdigest()[:12]

    manifest = {
        "status": "VERIFIED_FROZEN",
        "protocol_doc": "PROTOCOL.md",
        "train_samples": len(train_indices),
        "train_hash": train_hash,
        "test_samples": len(test_indices),
        "test_hash": test_hash,
        "protocol_seeds": list(PROTOCOL_SEEDS),
    }

    manifest_path = Path(output_dir) / "frozen_protocol.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("Frozen protocol verified (spec in PROTOCOL.md):")
    logger.info("  • Train split: %d samples (SHA: %s)", len(train_indices), train_hash)
    logger.info("  • Test split:  %d samples (SHA: %s)", len(test_indices), test_hash)
    logger.info("  • Protocol seeds: %s", PROTOCOL_SEEDS)
    logger.info("  • Manifest: %s", manifest_path)
    logger.info("============================================================\n")


def setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_file = Path(output_dir) / "experiment.log"

    logger = logging.getLogger("jepa_experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_jepa(
    model: JEPA,
    train_loader: torch.utils.data.DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    logger: logging.Logger,
    seed: int,
) -> JEPA:

    optimizer = optim.AdamW(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()),
        lr=lr,
    )
    model.train()

    logger.info("[Seed %d] Training JEPA for %d epochs...", seed, epochs)
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for images, _ in train_loader:
            images = images.to(device)
            context, target = split_context_target(images)

            pred_latent, target_latent = model(context, target)
            loss = F.mse_loss(pred_latent, target_latent)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.update_target_encoder()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        logger.info("[Seed %d] Epoch %d/%d - Avg Training MSE: %.4f", seed, epoch, epochs, avg_loss)

    return model


def evaluate_interventions_multibatch(
    model: JEPA,
    eval_loader: torch.utils.data.DataLoader,
    top_k: int,
    seed: int,
    device: torch.device,
    logger: logging.Logger,
) -> list[dict]:

    model.eval()

    all_context_latents = []
    total_base_loss = 0.0
    total_elements = 0

    with torch.no_grad():
        for images, _ in eval_loader:
            images = images.to(device)
            context, target = split_context_target(images)
            pred_latent, target_latent = model(context, target)

            loss = F.mse_loss(pred_latent, target_latent, reduction="sum")
            total_base_loss += loss.item()
            total_elements += pred_latent.numel()

            c_latent = model.encode(context)
            all_context_latents.append(c_latent.cpu())

    base_loss = total_base_loss / max(1, total_elements)
    aggregated_latents = torch.cat(all_context_latents, dim=0)

    intervener = CausalIntervention(aggregated_latents, seed=seed)
    directions, singular_values, explained_variances = intervener.extract_top_k_directions(k=top_k)

    logger.info(
        "[Seed %d] Extracted top-%d SVD directions from %d test samples.",
        seed,
        len(directions),
        len(aggregated_latents),
    )
    for rank, (sv, ev) in enumerate(zip(singular_values, explained_variances), start=1):
        logger.info(
            "  • SVD Rank %d: Singular Value = %.2f, Variance Explained = %.2f%%",
            rank,
            sv.item(),
            ev.item() * 100.0,
        )

    operations = [
        "Ablation (Projection-Out)",
        "Clamping (Mean-Projection)",
        "Amplification (2x)",
        "Swapping",
    ]

    records = []

    def evaluate_with_hook(
        direction: torch.Tensor, op_name: str, op_seed: int
    ) -> float:
        def modifying_hook(module, inputs, output):
            if op_name == "Ablation (Projection-Out)":
                return intervener.projection_out(direction, activations=output)
            elif op_name == "Clamping (Mean-Projection)":
                return intervener.clamping_to_mean(direction, activations=output)
            elif op_name == "Amplification (2x)":
                return intervener.amplification(direction, multiplier=2.0, activations=output)
            elif op_name == "Swapping":
                return intervener.swapping(direction, activations=output, seed=op_seed)
            raise ValueError(f"Unknown operation: {op_name}")

        handle = model.context_encoder[-1].register_forward_hook(modifying_hook)
        total_eval_loss = 0.0
        total_eval_elements = 0

        with torch.no_grad():
            for images, _ in eval_loader:
                images = images.to(device)
                context, target = split_context_target(images)
                pred_latent, target_latent = model(context, target)
                loss = F.mse_loss(pred_latent, target_latent, reduction="sum")
                total_eval_loss += loss.item()
                total_eval_elements += pred_latent.numel()

        handle.remove()
        return total_eval_loss / max(1, total_eval_elements)


    for r_idx in range(len(directions)):
        rank = r_idx + 1
        feature_dir = directions[r_idx]
        random_dir = intervener.get_matched_random_control(
            feature_dir, seed=seed + 1000 + rank
        )
        var_ratio = explained_variances[r_idx].item()

        for op_name in operations:
            feature_loss = evaluate_with_hook(
                feature_dir, op_name, op_seed=seed + rank
            )
            feature_effect = feature_loss - base_loss

            control_loss = evaluate_with_hook(
                random_dir, op_name, op_seed=seed + 500 + rank
            )
            control_effect = control_loss - base_loss

            causal_delta = feature_effect - control_effect

            records.append({
                "seed": seed,
                "svd_rank": rank,
                "explained_variance": var_ratio,
                "Intervention": op_name,
                "Base Loss": base_loss,
                "Feature Loss": feature_loss,
                "Feature Effect Size": feature_effect,
                "Control Loss": control_loss,
                "Control Effect Size": control_effect,
                "Causal Delta": causal_delta,
            })

    return records


def run_experiment(args: argparse.Namespace) -> None:
    logger = setup_logger(args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info("Executing experiment on compute device: %s", device)

    seeds = args.seeds
    logger.info("Initiating multi-seed protocol across %d seeds: %s", len(seeds), seeds)

    all_seed_records = []
    eval_loader = get_standard_test_loader(
        root=args.data_root,
        batch_size=args.batch_size,
        max_test_samples=PROTOCOL_TEST_SAMPLES,
        use_fixed_indices=True,
    )

    for s_idx, seed in enumerate(seeds, start=1):
        logger.info("------------------------------------------------------------")
        logger.info("Processing Seed %d/%d (Seed Value: %d)", s_idx, len(seeds), seed)
        logger.info("------------------------------------------------------------")
        set_seed(seed)

        train_loader, _ = get_mnist_loaders(
            root=args.data_root,
            batch_size=args.batch_size,
            seed=seed,
            use_fixed_indices=True,
        )

        model = JEPA(latent_dim=args.latent_dim).to(device)
        model = train_jepa(
            model=model,
            train_loader=train_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            logger=logger,
            seed=seed,
        )

        seed_records = evaluate_interventions_multibatch(
            model=model,
            eval_loader=eval_loader,
            top_k=args.top_k,
            seed=seed,
            device=device,
            logger=logger,
        )
        all_seed_records.extend(seed_records)

    raw_df = pd.DataFrame(all_seed_records)
    raw_csv_path = Path(args.output_dir) / "intervention_table_raw.csv"
    raw_df.to_csv(raw_csv_path, index=False)
    logger.info("Saved raw multi-seed intervention table → %s", raw_csv_path)

    summary_rows = []
    for (rank, op_name), group in raw_df.groupby(["svd_rank", "Intervention"], sort=False):
        summary_rows.append({
            "SVD Rank": rank,
            "Explained Var (%)": f"{group['explained_variance'].mean() * 100:.2f} ± {group['explained_variance'].std() * 100:.2f}",
            "Intervention": op_name,
            "Base Loss": f"{group['Base Loss'].mean():.4f} ± {group['Base Loss'].std():.4f}",
            "Feature Loss": f"{group['Feature Loss'].mean():.4f} ± {group['Feature Loss'].std():.4f}",
            "Feature Effect Size": f"{group['Feature Effect Size'].mean():.4f} ± {group['Feature Effect Size'].std():.4f}",
            "Control Loss": f"{group['Control Loss'].mean():.4f} ± {group['Control Loss'].std():.4f}",
            "Control Effect Size": f"{group['Control Effect Size'].mean():.4f} ± {group['Control Effect Size'].std():.4f}",
            "Causal Delta": f"{group['Causal Delta'].mean():.4f} ± {group['Causal Delta'].std():.4f}",
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = Path(args.output_dir) / "intervention_table_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info("Saved multi-seed summary table → %s", summary_csv_path)

    rank1_df = raw_df[raw_df["svd_rank"] == 1]
    rank1_numeric = rank1_df.groupby("Intervention", as_index=False).agg({
        "Base Loss": ["mean", "std"],
        "Feature Loss": ["mean", "std"],
        "Feature Effect Size": ["mean", "std"],
        "Control Loss": ["mean", "std"],
        "Control Effect Size": ["mean", "std"],
        "Causal Delta": ["mean", "std"],
    })
    rank1_numeric.columns = [
        "Intervention" if col[1] == "" else f"{col[0]} Std" if col[1] == "std" else col[0]
        for col in rank1_numeric.columns
    ]
    primary_csv_path = Path(args.output_dir) / "intervention_table.csv"
    rank1_numeric.to_csv(primary_csv_path, index=False)
    logger.info("Saved primary Rank-1 table → %s", primary_csv_path)

    spectrum_df = (
        raw_df.groupby("svd_rank", as_index=False)
        .agg(
            explained_var_mean=("explained_variance", "mean"),
            explained_var_std=("explained_variance", "std"),
        )
        .rename(columns={"svd_rank": "SVD Rank"})
    )
    spectrum_csv_path = Path(args.output_dir) / "svd_variance_spectrum.csv"
    spectrum_df.to_csv(spectrum_csv_path, index=False)
    logger.info("Saved SVD variance spectrum → %s", spectrum_csv_path)

    logger.info("\n============================================================")
    logger.info("FINAL STATISTICAL REPORT (RANK 1 PRINCIPAL SVD DIRECTION)")
    logger.info("============================================================")
    rank1_summary = summary_df[summary_df["SVD Rank"] == 1].drop(columns=["SVD Rank"])
    logger.info("\n%s\n", rank1_summary.to_string(index=False))

    logger.info("============================================================")
    logger.info("TOP-5 SVD DECAY SPECTRUM")
    logger.info("============================================================")
    decay_summary = summary_df[summary_df["Intervention"] == "Ablation (Projection-Out)"][
        ["SVD Rank", "Explained Var (%)", "Causal Delta"]
    ]
    logger.info("\n%s\n", decay_summary.to_string(index=False))

    if not args.skip_plots:
        generate_plots(
            primary_csv_path=str(primary_csv_path),
            raw_csv_path=str(raw_csv_path),
            output_dir=os.path.join(args.output_dir, "plots"),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run causal interpretability experiments on a scaled-down JEPA model."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs per seed (default: 5).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(PROTOCOL_SEEDS),
        help=f"List of random seeds for multi-seed averaging (default: {list(PROTOCOL_SEEDS)}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training and evaluation (default: 128).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to store logs, CSV tables, and plots (default: 'outputs').",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of principal SVD directions to evaluate (default: 5).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for AdamW optimizer (default: 1e-3).",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=64,
        help="Latent bottleneck dimension (default: 64).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root path for datasets and split files (default: 'data').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use ('cuda' or 'cpu', default: 'cuda' if available).",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation if only metrics/tables are needed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run_experiment(cli_args)

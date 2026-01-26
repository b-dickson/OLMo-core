"""
Chinchilla Scaling Law Fitting and Analysis

Fits the Chinchilla scaling law: L(N, D) = E + A/N^alpha + B/D^beta

Where:
- L: Loss
- N: Number of parameters
- D: Number of training tokens
- E: Irreducible loss (entropy of the data)
- A, alpha: Parameters governing model size scaling
- B, beta: Parameters governing data scaling

Usage:
    python scripts/analysis/fit_scaling_laws.py --metrics-dir ./results --output-dir ./analysis
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from scipy.optimize import curve_fit

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    log.warning("scipy not available, curve fitting will not work")

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    log.warning("matplotlib not available, plotting will not work")


# Model size specifications (non-embedding parameters)
MODEL_SIZES = {
    "60M": 60_000_000,
    "100M": 100_000_000,
    "190M": 190_000_000,
    "370M": 370_000_000,
    "600M": 600_000_000,
    "760M": 760_000_000,
    "1B": 1_000_000_000,
}

# Training sizes used to fit scaling law
TRAIN_SIZES = ["60M", "100M", "190M", "370M", "760M"]

# Test sizes for validation (interpolation and extrapolation)
TEST_SIZES = ["600M", "1B"]

# Chinchilla multiples
CHINCHILLA_MULTIPLES = [0.5, 1.0, 2.0, 4.0]


def chinchilla_tokens(n_params: int, multiple: float = 1.0) -> int:
    """Calculate Chinchilla-optimal tokens for a given model size."""
    return int(20 * n_params * multiple)


def chinchilla_loss(ND, E, A, alpha, B, beta):
    """
    Chinchilla scaling law: L(N, D) = E + A/N^alpha + B/D^beta

    Args:
        ND: Tuple of (N, D) arrays - number of parameters and tokens
        E: Irreducible loss
        A: Model size coefficient
        alpha: Model size exponent
        B: Data size coefficient
        beta: Data size exponent

    Returns:
        Predicted loss values
    """
    N, D = ND
    return E + A / np.power(N, alpha) + B / np.power(D, beta)


def fit_scaling_law(df: pd.DataFrame) -> dict:
    """
    Fit Chinchilla scaling law to training data.

    Args:
        df: DataFrame with columns 'num_params', 'tokens', 'loss'

    Returns:
        Dictionary with fitted parameters {E, A, alpha, B, beta}
    """
    if not HAS_SCIPY:
        raise RuntimeError("scipy is required for curve fitting")

    N = df["num_params"].values.astype(float)
    D = df["tokens"].values.astype(float)
    L = df["loss"].values.astype(float)

    # Initial guesses based on Chinchilla paper
    # E ~ 1.69, A ~ 406.4, alpha ~ 0.34, B ~ 410.7, beta ~ 0.28
    p0 = [1.69, 406.4, 0.34, 410.7, 0.28]
    bounds = ([0, 0, 0, 0, 0], [10, 1e6, 1, 1e6, 1])

    try:
        popt, pcov = curve_fit(
            lambda ND, E, A, alpha, B, beta: chinchilla_loss(ND, E, A, alpha, B, beta),
            (N, D),
            L,
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )

        # Calculate standard errors from covariance matrix
        perr = np.sqrt(np.diag(pcov))

        return {
            "E": popt[0],
            "A": popt[1],
            "alpha": popt[2],
            "B": popt[3],
            "beta": popt[4],
            "E_err": perr[0],
            "A_err": perr[1],
            "alpha_err": perr[2],
            "B_err": perr[3],
            "beta_err": perr[4],
        }
    except Exception as e:
        log.error(f"Curve fitting failed: {e}")
        raise


def evaluate_predictions(params: dict, test_df: pd.DataFrame) -> dict:
    """
    Compare predicted vs observed loss on held-out test set.

    Args:
        params: Fitted scaling law parameters
        test_df: DataFrame with columns 'num_params', 'tokens', 'loss'

    Returns:
        Dictionary with evaluation metrics and predictions
    """
    N = test_df["num_params"].values.astype(float)
    D = test_df["tokens"].values.astype(float)
    L_observed = test_df["loss"].values.astype(float)

    L_predicted = chinchilla_loss(
        (N, D), params["E"], params["A"], params["alpha"], params["B"], params["beta"]
    )

    mse = np.mean((L_predicted - L_observed) ** 2)
    mae = np.mean(np.abs(L_predicted - L_observed))
    mape = np.mean(np.abs((L_predicted - L_observed) / L_observed)) * 100

    return {
        "mse": mse,
        "mae": mae,
        "mape": mape,
        "predicted": L_predicted.tolist(),
        "observed": L_observed.tolist(),
        "residuals": (L_predicted - L_observed).tolist(),
    }


def load_metrics(metrics_dir: Path, attention_type: str) -> Optional[pd.DataFrame]:
    """
    Load metrics files for a given attention type.

    Args:
        metrics_dir: Directory containing metrics pickle files
        attention_type: One of 'full', 'sliding', 'linear'

    Returns:
        DataFrame with columns: size, num_params, tokens, loss, chinchilla_multiple
    """
    ladder_dir = metrics_dir / f"attn-scaling-{attention_type}"
    if not ladder_dir.exists():
        log.warning(f"Metrics directory not found: {ladder_dir}")
        return None

    records = []
    for size_name in list(TRAIN_SIZES) + list(TEST_SIZES):
        for mult in CHINCHILLA_MULTIPLES:
            # Try to find metrics file
            # Format depends on how metrics were saved
            metrics_file = ladder_dir / f"metrics_{size_name}.pkl"

            if metrics_file.exists():
                try:
                    df = pd.read_pickle(metrics_file)
                    # Get final loss
                    if "train/CE loss" in df.columns:
                        final_loss = df["train/CE loss"].iloc[-1]
                    elif "loss" in df.columns:
                        final_loss = df["loss"].iloc[-1]
                    else:
                        log.warning(f"No loss column found in {metrics_file}")
                        continue

                    records.append(
                        {
                            "size": size_name,
                            "num_params": MODEL_SIZES[size_name],
                            "tokens": chinchilla_tokens(MODEL_SIZES[size_name], mult),
                            "chinchilla_multiple": mult,
                            "loss": final_loss,
                            "attention_type": attention_type,
                        }
                    )
                except Exception as e:
                    log.warning(f"Failed to load {metrics_file}: {e}")

    if not records:
        return None

    return pd.DataFrame(records)


def plot_scaling_laws(
    results: dict,
    output_dir: Path,
    attention_types: list = ["full", "sliding", "gated_deltanet", "deltanet", "mamba2", "rwkv7"],
):
    """
    Generate comparison plots for scaling laws across attention types.

    Args:
        results: Dictionary with results for each attention type
        output_dir: Directory to save plots
        attention_types: List of attention types to plot
    """
    if not HAS_MATPLOTLIB:
        log.warning("matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Colors for different attention types
    colors = {
        "full": "blue",
        "sliding": "green",
        "gated_deltanet": "red",
        "deltanet": "orange",
        "mamba2": "purple",
        "rwkv7": "brown",
    }

    # Plot 1: Loss vs Parameters (fixed tokens = 1x Chinchilla)
    fig, ax = plt.subplots(figsize=(10, 6))
    for attn in attention_types:
        if attn not in results or "params" not in results[attn]:
            continue

        params = results[attn]["params"]
        N_range = np.logspace(7, 10, 100)
        D_fixed = 20 * N_range  # 1x Chinchilla

        L = chinchilla_loss(
            (N_range, D_fixed), params["E"], params["A"], params["alpha"], params["B"], params["beta"]
        )
        ax.loglog(N_range, L, color=colors.get(attn, "gray"), label=f"{attn} (alpha={params['alpha']:.3f})")

    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Loss")
    ax.set_title("Scaling with Model Size (1x Chinchilla tokens)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "scaling_vs_params.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: Loss vs Tokens (fixed params = 370M)
    fig, ax = plt.subplots(figsize=(10, 6))
    N_fixed = MODEL_SIZES["370M"]
    D_range = np.logspace(9, 12, 100)

    for attn in attention_types:
        if attn not in results or "params" not in results[attn]:
            continue

        params = results[attn]["params"]
        L = chinchilla_loss(
            (N_fixed, D_range), params["E"], params["A"], params["alpha"], params["B"], params["beta"]
        )
        ax.loglog(D_range, L, color=colors.get(attn, "gray"), label=f"{attn} (beta={params['beta']:.3f})")

    ax.set_xlabel("Number of Tokens")
    ax.set_ylabel("Loss")
    ax.set_title("Scaling with Data Size (370M params)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "scaling_vs_tokens.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 3: Parameter comparison table
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    table_data = [["Attention Type", "E", "alpha", "beta", "Test MAPE (%)"]]
    for attn in attention_types:
        if attn not in results or "params" not in results[attn]:
            continue
        params = results[attn]["params"]
        test_mape = results[attn].get("test_eval", {}).get("mape", float("nan"))
        table_data.append(
            [attn, f"{params['E']:.3f}", f"{params['alpha']:.3f}", f"{params['beta']:.3f}", f"{test_mape:.2f}"]
        )

    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    ax.set_title("Scaling Law Parameters Comparison", fontsize=14, pad=20)
    fig.savefig(output_dir / "parameter_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fit Chinchilla scaling laws to attention variant experiments")
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        required=True,
        help="Directory containing metrics files from ladder experiments",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./scaling_analysis"),
        help="Output directory for analysis results and plots",
    )
    parser.add_argument(
        "--attention-types",
        nargs="+",
        default=["full", "sliding", "gated_deltanet"],
        help="Attention types to analyze (options: full, sliding, gated_deltanet, deltanet, mamba2, rwkv7)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for attn_type in args.attention_types:
        log.info(f"\n{'='*60}")
        log.info(f"Processing attention type: {attn_type}")
        log.info(f"{'='*60}")

        # Load metrics
        df = load_metrics(args.metrics_dir, attn_type)
        if df is None or df.empty:
            log.warning(f"No metrics found for {attn_type}, skipping")
            continue

        # Split into train and test
        train_df = df[df["size"].isin(TRAIN_SIZES)]
        test_df = df[df["size"].isin(TEST_SIZES)]

        log.info(f"Training points: {len(train_df)}")
        log.info(f"Test points: {len(test_df)}")

        if len(train_df) < 5:
            log.warning(f"Not enough training points for {attn_type}, need at least 5")
            continue

        # Fit scaling law
        try:
            params = fit_scaling_law(train_df)
            log.info(f"Fitted parameters for {attn_type}:")
            log.info(f"  E (irreducible) = {params['E']:.4f} +/- {params['E_err']:.4f}")
            log.info(f"  A = {params['A']:.4f} +/- {params['A_err']:.4f}")
            log.info(f"  alpha = {params['alpha']:.4f} +/- {params['alpha_err']:.4f}")
            log.info(f"  B = {params['B']:.4f} +/- {params['B_err']:.4f}")
            log.info(f"  beta = {params['beta']:.4f} +/- {params['beta_err']:.4f}")

            results[attn_type] = {"params": params, "train_df": train_df.to_dict()}

            # Evaluate on training set
            train_eval = evaluate_predictions(params, train_df)
            log.info(f"Training set evaluation:")
            log.info(f"  MSE = {train_eval['mse']:.6f}")
            log.info(f"  MAE = {train_eval['mae']:.6f}")
            log.info(f"  MAPE = {train_eval['mape']:.2f}%")
            results[attn_type]["train_eval"] = train_eval

            # Evaluate on test set if available
            if len(test_df) > 0:
                test_eval = evaluate_predictions(params, test_df)
                log.info(f"Test set evaluation:")
                log.info(f"  MSE = {test_eval['mse']:.6f}")
                log.info(f"  MAE = {test_eval['mae']:.6f}")
                log.info(f"  MAPE = {test_eval['mape']:.2f}%")
                results[attn_type]["test_eval"] = test_eval
                results[attn_type]["test_df"] = test_df.to_dict()

        except Exception as e:
            log.error(f"Failed to fit scaling law for {attn_type}: {e}")
            continue

    # Save results
    results_file = args.output_dir / "scaling_law_results.json"

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    with open(results_file, "w") as f:
        json.dump(convert_numpy(results), f, indent=2)
    log.info(f"Results saved to {results_file}")

    # Generate plots
    plot_scaling_laws(results, args.output_dir, args.attention_types)

    # Print summary table
    print("\n" + "=" * 80)
    print("SCALING LAW COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Attention':<12} {'E':>8} {'alpha':>8} {'beta':>8} {'Train MAPE':>12} {'Test MAPE':>12}")
    print("-" * 80)
    for attn_type in args.attention_types:
        if attn_type not in results:
            continue
        params = results[attn_type]["params"]
        train_mape = results[attn_type].get("train_eval", {}).get("mape", float("nan"))
        test_mape = results[attn_type].get("test_eval", {}).get("mape", float("nan"))
        print(
            f"{attn_type:<12} {params['E']:>8.4f} {params['alpha']:>8.4f} {params['beta']:>8.4f} "
            f"{train_mape:>11.2f}% {test_mape:>11.2f}%"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()

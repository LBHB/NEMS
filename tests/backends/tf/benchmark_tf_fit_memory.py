"""Benchmark repeated-fit memory use for the TF custom loop vs direct Keras fit.

This is a standalone benchmark script, not a collected pytest test. It exists
to answer one narrow question: after repeated fits on the same large payload,
how much process memory still accumulates after explicit TensorFlow cleanup?

The model architecture is the freemoving stack from
`nems_lbhb/projects/freemoving/dlc_simp_model_fitscript.py`, but with the
loader-time placeholders instantiated to concrete sizes so it can run without
the full xform pipeline:

- audio input channels: 36
- DLC input channels: 6
- output channels: 218

Examples
--------
Run both modes in isolated subprocesses:

    conda run -n psi_nems2 python tests/backends/tf/benchmark_tf_fit_memory.py --mode both

Run only the current NEMS custom loop:

    conda run -n psi_nems2 python tests/backends/tf/benchmark_tf_fit_memory.py --mode custom

Use a smaller smoke-test payload:

    conda run -n psi_nems2 python tests/backends/tf/benchmark_tf_fit_memory.py \\
        --mode both --samples 8 --time-bins 100 --repeats 2 --epochs 1 --batch-size 2
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import psutil

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[3]
NEMS_DB_ROOT = REPO_ROOT.parent / "nems_db"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if NEMS_DB_ROOT.exists() and str(NEMS_DB_ROOT) not in sys.path:
    sys.path.insert(0, str(NEMS_DB_ROOT))

import tensorflow as tf

MiB = 1024 * 1024
PROCESS = psutil.Process(os.getpid())

warnings.filterwarnings(
    "ignore",
    message="Unable to find acceptable character detection dependency*",
)
logging.getLogger("nems.backends.tf.backend").setLevel(logging.WARNING)
logging.getLogger("nems.models.LN").setLevel(logging.ERROR)

import nems_lbhb.plugins.free_kw  # Registers freemoving keyword wrappers.
import matplotlib.pyplot as plt
from nems import Model
from nems.backends.tf.backend import TensorFlowBackend
from nems.backends.tf.cost import get_cost
from nems.models.dataset import DataSet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("custom", "keras", "both"),
        default="both",
        help="Benchmark the NEMS custom loop, direct keras.Model.fit, or both.",
    )
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--time-bins", type=int, default=1000)
    parser.add_argument("--input-channels", type=int, default=36)
    parser.add_argument("--dlc-channels", type=int, default=6)
    parser.add_argument("--output-channels", type=int, default=218)
    parser.add_argument("--acount", type=int, default=40)
    parser.add_argument("--l2count", type=int, default=60)
    parser.add_argument("--dcount", type=int, default=16)
    parser.add_argument("--firlen", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--grad-clipnorm", type=float, default=1.0)
    parser.add_argument("--cost-function", default="squared_error")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail immediately if TensorFlow cannot see a GPU device.",
    )
    parser.add_argument(
        "--retain-backend",
        action="store_true",
        help="Keep the TF backend on the returned model for custom mode.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path where the benchmark payload will be written as JSON.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        help="Optional path where an RSS-vs-fit line plot will be written.",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _payload_mb(inputs: dict[str, np.ndarray], target: np.ndarray) -> float:
    total_bytes = sum(array.nbytes for array in inputs.values()) + target.nbytes
    return total_bytes / MiB


def _gpu_snapshot() -> dict[str, float | None]:
    if not tf.config.list_physical_devices("GPU"):
        return {"gpu_current_mb": None, "gpu_peak_mb": None}
    try:
        info = tf.config.experimental.get_memory_info("GPU:0")
        return {
            "gpu_current_mb": info["current"] / MiB,
            "gpu_peak_mb": info["peak"] / MiB,
        }
    except Exception:
        return {"gpu_current_mb": None, "gpu_peak_mb": None}


def _reset_gpu_peak() -> None:
    if not tf.config.list_physical_devices("GPU"):
        return
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception:
        return


def _memory_snapshot() -> dict[str, float | None]:
    snapshot = {"rss_mb": PROCESS.memory_info().rss / MiB}
    snapshot.update(_gpu_snapshot())
    return snapshot


def collect_runtime_info() -> dict[str, object]:
    gpu_devices = tf.config.list_physical_devices("GPU")
    build_info = tf.sysconfig.get_build_info()
    return {
        "host": os.uname().nodename,
        "tf_version": tf.__version__,
        "built_with_cuda": tf.test.is_built_with_cuda(),
        "visible_gpus": [device.name for device in gpu_devices],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_version": build_info.get("cuda_version"),
        "cudnn_version": build_info.get("cudnn_version"),
    }


def ensure_gpu_available() -> None:
    runtime = collect_runtime_info()
    gpu_devices = runtime["visible_gpus"]
    if gpu_devices:
        return

    lines = [
        "GPU is required for this benchmark, but TensorFlow cannot see one.",
        f"host={runtime['host']}",
        f"tf_version={runtime['tf_version']}",
        f"built_with_cuda={runtime['built_with_cuda']}",
        f"visible_gpus={gpu_devices}",
        f"CUDA_VISIBLE_DEVICES={runtime['cuda_visible_devices']}",
        f"cuda_version={runtime['cuda_version']}",
        f"cudnn_version={runtime['cudnn_version']}",
    ]

    if shutil.which("nvidia-smi") is not None:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            lines.append(f"nvidia-smi -L: {result.stdout.strip()}")
        else:
            stderr = result.stderr.strip() or result.stdout.strip() or "unknown error"
            lines.append(f"nvidia-smi -L failed: {stderr}")
    else:
        lines.append("nvidia-smi not found on PATH")

    raise RuntimeError("\n".join(lines))


def _cleanup_tf() -> None:
    tf.keras.backend.clear_session()
    gc.collect()


def make_benchmark_keyword(args: argparse.Namespace) -> str:
    reg = ".l2:4"
    return (
        f"wcdl.{args.dlc_channels}x{args.dcount}.i.s{reg}-"
        f"relus.{args.dcount}-"
        f"wcs.{args.dcount}x{args.dcount}{reg}-"
        f"relus.{args.dcount}-"
        f"wc.{args.input_channels}x1x{args.acount}.i{reg}-"
        f"fir.{args.firlen}x1x{args.acount}-"
        f"relu.{args.acount}-"
        f"wc.{args.acount}x1x{args.l2count}{reg}-"
        f"fir.{args.firlen}x1x{args.l2count}-"
        f"relu.{args.l2count}-"
        f"wc.{args.l2count}x{args.l2count}-"
        f"relu.{args.l2count}{reg}-"
        f"wc.{args.l2count}x{args.l2count}-"
        f"relu.{args.l2count}{reg}-"
        f"wc.{args.l2count}x{args.output_channels}{reg}-"
        f"stategain.{args.dcount}x{args.output_channels}{reg}-"
        f"relu.{args.output_channels}.o.s"
    )


def make_synthetic_data(
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(args.seed)
    inputs = {
        "input": rng.normal(
            size=(args.samples, args.time_bins, args.input_channels)
        ).astype(np.float32),
        "dlc": rng.normal(
            size=(args.samples, args.time_bins, args.dlc_channels)
        ).astype(np.float32),
    }
    target = rng.normal(
        size=(args.samples, args.time_bins, args.output_channels)
    ).astype(np.float32)
    return inputs, target


def _fit_history_last_loss(fitted_model: Model) -> float:
    history = fitted_model.results.misc["misc"]["TensorFlow History"]
    return float(history["loss"][-1])


def run_custom_fit(
    args: argparse.Namespace,
    inputs: dict[str, np.ndarray],
    target: np.ndarray,
    iteration_seed: int,
) -> float:
    np.random.seed(iteration_seed)
    tf.random.set_seed(iteration_seed)
    model = Model.from_keywords(make_benchmark_keyword(args))
    fitted = model.fit(
        inputs,
        target,
        batch_size=args.batch_size,
        backend="tf",
        verbose=0,
        retain_backend=args.retain_backend,
        fitter_options={
            "cost_function": args.cost_function,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "grad_clipnorm": args.grad_clipnorm,
            "early_stopping_tolerance": 0,
            "shuffle": args.shuffle,
        },
    )
    loss = _fit_history_last_loss(fitted)
    del fitted
    del model
    _cleanup_tf()
    return loss


def run_direct_keras_fit(
    args: argparse.Namespace,
    inputs: dict[str, np.ndarray],
    target: np.ndarray,
    iteration_seed: int,
) -> float:
    np.random.seed(iteration_seed)
    tf.random.set_seed(iteration_seed)

    model = Model.from_keywords(make_benchmark_keyword(args))
    peek_count = min(args.batch_size, args.samples)
    peek_inputs = {key: value[:peek_count] for key, value in inputs.items()}
    _ = model.evaluate(peek_inputs, batch_size=peek_count, use_existing_maps=False)

    data = DataSet(inputs, target=target)
    backend = TensorFlowBackend(
        model.copy(),
        data,
        verbose=0,
        eval_kwargs={"batch_size": args.batch_size},
    )
    backend.model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate,
            clipnorm=args.grad_clipnorm,
        ),
        loss=get_cost(args.cost_function),
    )
    history = backend.model.fit(
        inputs,
        target,
        batch_size=args.batch_size,
        epochs=args.epochs,
        shuffle=args.shuffle,
        verbose=0,
    )
    loss = float(history.history["loss"][-1])
    del history
    del backend
    del data
    del model
    _cleanup_tf()
    return loss


def run_iteration(
    mode: str,
    args: argparse.Namespace,
    inputs: dict[str, np.ndarray],
    target: np.ndarray,
    index: int,
    phase: str,
) -> dict[str, float | int | str | None]:
    runner = run_custom_fit if mode == "custom" else run_direct_keras_fit
    _reset_gpu_peak()
    before = _memory_snapshot()
    start = time.perf_counter()
    loss = runner(args, inputs, target, iteration_seed=args.seed + index)
    elapsed_s = time.perf_counter() - start
    after = _memory_snapshot()
    row = {
        "phase": phase,
        "iteration": index,
        "loss": loss,
        "elapsed_s": elapsed_s,
        "rss_before_mb": before["rss_mb"],
        "rss_after_cleanup_mb": after["rss_mb"],
        "gpu_before_mb": before["gpu_current_mb"],
        "gpu_after_cleanup_mb": after["gpu_current_mb"],
        "gpu_peak_mb": after["gpu_peak_mb"],
    }
    print(
        f"[{mode}] {phase} {index + 1}: "
        f"loss={loss:.4f}, elapsed={elapsed_s:.1f}s, "
        f"rss_after_cleanup={after['rss_mb']:.1f} MiB",
        flush=True,
    )
    return row


def summarize_results(
    mode: str,
    args: argparse.Namespace,
    inputs: dict[str, np.ndarray],
    target: np.ndarray,
    baseline: dict[str, float | None],
    rows: list[dict[str, float | int | str | None]],
) -> dict[str, float | int | None]:
    warmup_rows = [row for row in rows if row["phase"] == "warmup"]
    fit_rows = [row for row in rows if row["phase"] == "fit"]
    rss_after_cleanup = np.array(
        [float(row["rss_after_cleanup_mb"]) for row in fit_rows], dtype=float
    )

    if warmup_rows:
        start_mb = float(warmup_rows[-1]["rss_after_cleanup_mb"])
    else:
        start_mb = float(baseline["rss_mb"])
    final_mb = float(rss_after_cleanup[-1]) if len(rss_after_cleanup) else start_mb
    peak_mb = float(rss_after_cleanup.max()) if len(rss_after_cleanup) else start_mb
    slope_mb_per_fit = (
        float(np.polyfit(np.arange(len(rss_after_cleanup)), rss_after_cleanup, 1)[0])
        if len(rss_after_cleanup) >= 2
        else 0.0
    )

    gpu_peaks = [
        float(row["gpu_peak_mb"]) for row in fit_rows if row["gpu_peak_mb"] is not None
    ]
    losses = [float(row["loss"]) for row in fit_rows]
    return {
        "mode": mode,
        "dataset_payload_mb": _payload_mb(inputs, target),
        "rss_after_data_mb": float(baseline["rss_mb"]),
        "rss_start_mb": start_mb,
        "rss_final_mb": final_mb,
        "rss_peak_after_cleanup_mb": peak_mb,
        "rss_delta_mb": final_mb - start_mb,
        "rss_slope_mb_per_fit": slope_mb_per_fit,
        "gpu_peak_mb": max(gpu_peaks) if gpu_peaks else None,
        "last_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
        "max_loss": max(losses) if losses else None,
        "epochs": args.epochs,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "batch_size": args.batch_size,
    }


def print_summary(summary: dict[str, float | int | None]) -> None:
    print(
        f"[{summary['mode']}] payload={summary['dataset_payload_mb']:.1f} MiB, "
        f"rss_start={summary['rss_start_mb']:.1f} MiB, "
        f"rss_final={summary['rss_final_mb']:.1f} MiB, "
        f"delta={summary['rss_delta_mb']:.1f} MiB, "
        f"slope={summary['rss_slope_mb_per_fit']:.2f} MiB/fit, "
        f"gpu_peak={summary['gpu_peak_mb'] if summary['gpu_peak_mb'] is not None else 'n/a'}",
        flush=True,
    )


def configure_plot_style() -> None:
    plt.rcdefaults()
    font_size = 20
    params = {
        "legend.fontsize": font_size - 12,
        "figure.figsize": (14, 10),
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "font.family": "Calibri",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 5,
        "xtick.major.size": 18,
        "ytick.major.size": 18,
        "xtick.major.width": 5,
        "ytick.major.width": 5,
        "xtick.minor.width": 3,
        "ytick.minor.width": 3,
        "xtick.minor.size": 5,
        "ytick.minor.size": 5,
    }
    plt.rcParams.update(params)


def resolve_plot_output(args: argparse.Namespace) -> Path | None:
    if args.plot_output is not None:
        return args.plot_output
    if args.worker:
        return None
    if args.json_output is not None:
        return args.json_output.with_suffix(".png")
    return None


def plot_memory_results(
    results: list[dict[str, object]],
    plot_output: Path,
) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(2, 2, sharex="col")
    ax_rss = axes[0, 0]
    ax_delta = axes[0, 1]
    ax_gpu = axes[1, 0]
    ax_cumulative = axes[1, 1]
    colors = {"custom": "#1f6f8b", "keras": "#c65d3b"}

    max_fit = 1
    any_gpu_data = False
    for result in results:
        summary = result["summary"]
        rows = result["rows"]
        fit_rows = [row for row in rows if row["phase"] == "fit"]
        if not fit_rows:
            continue

        fit_numbers = [int(row["iteration"]) + 1 for row in fit_rows]
        color = colors.get(str(summary["mode"]))
        rss_before = [float(row["rss_before_mb"]) for row in fit_rows]
        rss_after = [float(row["rss_after_cleanup_mb"]) for row in fit_rows]
        rss_delta = [after - before for before, after in zip(rss_before, rss_after)]
        rss_cumulative = [
            after - float(summary["rss_start_mb"]) for after in rss_after
        ]
        mode = str(summary["mode"])
        slope = float(summary["rss_slope_mb_per_fit"])
        max_fit = max(max_fit, max(fit_numbers))

        ax_rss.axhline(
            float(summary["rss_start_mb"]),
            color=color,
            linestyle=":",
            linewidth=2,
            alpha=0.8,
        )
        ax_rss.plot(
            fit_numbers,
            rss_before,
            marker="o",
            linewidth=3,
            markersize=8,
            color=color,
            label=f"{mode} before",
        )
        ax_rss.plot(
            fit_numbers,
            rss_after,
            marker="s",
            linewidth=3,
            markersize=8,
            color=color,
            linestyle="--",
            label=f"{mode} after cleanup",
        )

        ax_delta.plot(
            fit_numbers,
            rss_delta,
            marker="o",
            linewidth=3,
            markersize=8,
            color=color,
            label=mode,
        )

        gpu_before = [
            float(row["gpu_before_mb"]) if row["gpu_before_mb"] is not None else np.nan
            for row in fit_rows
        ]
        gpu_after = [
            float(row["gpu_after_cleanup_mb"])
            if row["gpu_after_cleanup_mb"] is not None
            else np.nan
            for row in fit_rows
        ]
        gpu_peak = [
            float(row["gpu_peak_mb"]) if row["gpu_peak_mb"] is not None else np.nan
            for row in fit_rows
        ]
        if not np.all(np.isnan(gpu_peak)):
            any_gpu_data = True
            ax_gpu.plot(
                fit_numbers,
                gpu_before,
                marker="o",
                linewidth=3,
                markersize=8,
                color=color,
                label=f"{mode} before",
            )
            ax_gpu.plot(
                fit_numbers,
                gpu_after,
                marker="s",
                linewidth=3,
                markersize=8,
                color=color,
                linestyle="--",
                label=f"{mode} after cleanup",
            )
            ax_gpu.plot(
                fit_numbers,
                gpu_peak,
                marker="^",
                linewidth=3,
                markersize=8,
                color=color,
                linestyle="-.",
                label=f"{mode} peak",
            )

        ax_cumulative.plot(
            fit_numbers,
            rss_cumulative,
            marker="o",
            linewidth=3,
            markersize=8,
            color=color,
            label=f"{mode} ({slope:.2f} MiB/fit)",
        )

    payload_mb = float(results[0]["summary"]["dataset_payload_mb"])
    repeats = int(results[0]["summary"]["repeats"])
    epochs = int(results[0]["summary"]["epochs"])

    ax_rss.set_ylabel("RSS (MiB)")
    ax_rss.set_title("RSS before vs after cleanup")
    ax_rss.grid(axis="y", alpha=0.2, linewidth=2)
    ax_rss.legend(frameon=False)

    ax_delta.set_ylabel("Delta (MiB)")
    ax_delta.set_title("RSS retained per fit")
    ax_delta.axhline(0, color="black", linewidth=2, alpha=0.3)
    ax_delta.grid(axis="y", alpha=0.2, linewidth=2)
    ax_delta.legend(frameon=False)

    ax_gpu.set_xlabel("Fit number")
    ax_gpu.set_ylabel("GPU memory (MiB)")
    ax_gpu.set_title("GPU snapshots")
    if any_gpu_data:
        ax_gpu.grid(axis="y", alpha=0.2, linewidth=2)
        ax_gpu.legend(frameon=False)
    else:
        ax_gpu.text(
            0.5,
            0.5,
            "No GPU memory info available",
            ha="center",
            va="center",
            transform=ax_gpu.transAxes,
        )
        ax_gpu.set_xticks([])
        ax_gpu.set_yticks([])

    ax_cumulative.set_xlabel("Fit number")
    ax_cumulative.set_ylabel("Growth from start (MiB)")
    ax_cumulative.set_title("Cumulative RSS growth")
    ax_cumulative.axhline(0, color="black", linewidth=2, alpha=0.3)
    ax_cumulative.grid(axis="y", alpha=0.2, linewidth=2)
    ax_cumulative.legend(frameon=False)

    for axis in (ax_rss, ax_delta, ax_gpu, ax_cumulative):
        if max_fit <= 1:
            axis.set_xlim(0.75, 1.25)
        else:
            axis.set_xlim(1, max_fit)

    fig.suptitle(
        f"Repeated TF fit memory\npayload={payload_mb:.1f} MiB, repeats={repeats}, epochs={epochs}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plot_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_output, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {plot_output}", flush=True)


def run_worker(args: argparse.Namespace) -> dict[str, object]:
    inputs, target = make_synthetic_data(args)
    baseline = _memory_snapshot()
    runtime = collect_runtime_info()
    print(
        f"[{args.mode}] host={runtime['host']}, "
        f"visible_gpus={runtime['visible_gpus']}, "
        f"payload={_payload_mb(inputs, target):.1f} MiB, "
        f"warmup={args.warmup}, repeats={args.repeats}, epochs={args.epochs}, "
        f"batch_size={args.batch_size}",
        flush=True,
    )

    rows: list[dict[str, float | int | str | None]] = []
    for index in range(args.warmup):
        rows.append(run_iteration(args.mode, args, inputs, target, index, "warmup"))
    for index in range(args.repeats):
        rows.append(run_iteration(args.mode, args, inputs, target, index, "fit"))

    summary = summarize_results(args.mode, args, inputs, target, baseline, rows)
    payload = {
        "mode": args.mode,
        "config": {
            "samples": args.samples,
            "time_bins": args.time_bins,
            "input_channels": args.input_channels,
            "dlc_channels": args.dlc_channels,
            "output_channels": args.output_channels,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "learning_rate": args.learning_rate,
            "grad_clipnorm": args.grad_clipnorm,
            "shuffle": args.shuffle,
            "retain_backend": args.retain_backend,
        },
        "runtime": runtime,
        "summary": summary,
        "rows": rows,
    }
    print_summary(summary)
    return payload


def args_to_child_cli(args: argparse.Namespace, mode: str, json_output: Path) -> list[str]:
    cli = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        mode,
        "--worker",
        "--json-output",
        str(json_output),
        "--samples",
        str(args.samples),
        "--time-bins",
        str(args.time_bins),
        "--input-channels",
        str(args.input_channels),
        "--dlc-channels",
        str(args.dlc_channels),
        "--output-channels",
        str(args.output_channels),
        "--acount",
        str(args.acount),
        "--l2count",
        str(args.l2count),
        "--dcount",
        str(args.dcount),
        "--firlen",
        str(args.firlen),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--learning-rate",
        str(args.learning_rate),
        "--grad-clipnorm",
        str(args.grad_clipnorm),
        "--cost-function",
        args.cost_function,
        "--seed",
        str(args.seed),
    ]
    if args.require_gpu:
        cli.append("--require-gpu")
    if args.shuffle:
        cli.append("--shuffle")
    if args.retain_backend:
        cli.append("--retain-backend")
    return cli


def run_both_modes(args: argparse.Namespace) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for mode in ("custom", "keras"):
        with tempfile.NamedTemporaryFile(
            prefix=f"benchmark_tf_fit_memory_{mode}_",
            suffix=".json",
            delete=False,
        ) as handle:
            json_path = Path(handle.name)
        try:
            subprocess.run(args_to_child_cli(args, mode, json_path), check=True)
            results.append(json.loads(json_path.read_text()))
        finally:
            json_path.unlink(missing_ok=True)
    return results


def print_comparison(results: list[dict[str, object]]) -> None:
    print("\nComparison")
    print("==========")
    for result in results:
        print_summary(result["summary"])


def main() -> None:
    args = parse_args()
    if args.require_gpu:
        ensure_gpu_available()

    if args.mode == "both" and not args.worker:
        results = run_both_modes(args)
        if args.json_output is not None:
            args.json_output.write_text(json.dumps(results, indent=2))
        plot_output = resolve_plot_output(args)
        if plot_output is not None:
            plot_memory_results(results, plot_output)
        print_comparison(results)
        return

    payload = run_worker(args)
    if args.json_output is not None:
        args.json_output.write_text(json.dumps(payload, indent=2))
    plot_output = resolve_plot_output(args)
    if plot_output is not None:
        plot_memory_results([payload], plot_output)


if __name__ == "__main__":
    main()

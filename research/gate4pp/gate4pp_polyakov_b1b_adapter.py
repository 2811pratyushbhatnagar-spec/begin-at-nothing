"""
Gate 4pp B1b adapter for center-projected winding-line correlators.

This is a thin live-U export path. It does not modify sampler physics and does
not use the sampler's summary-only _run_chain(...).

It dynamically loads a local sampler file and uses its existing:
    _initialize_hot_no_flat(L, N, max_tries)
    _initialize_uniform_flux(L, N, hstar)
    _sweep(U, N, beta, small_steps)

At each measurement point, the live U is accumulated into the Polyakov-style /
winding-line correlator ensemble.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

import numpy as np

try:
    from .z9_polyakov_correlators import EnsemblePolyakovAccumulator, analyze_ensemble, make_json_safe
except ImportError:
    from z9_polyakov_correlators import EnsemblePolyakovAccumulator, analyze_ensemble, make_json_safe


def beta_file_label(beta: float) -> str:
    return f"beta_{beta:.4f}".replace(".", "p")


def load_sampler_module(path: Path) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(f"sampler file not found: {path}")
    spec = importlib.util.spec_from_file_location("b1b_zn3d_scaling_sampler_dynamic", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load sampler module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_sampler_api(sampler: ModuleType) -> None:
    required = ["_initialize_hot_no_flat", "_initialize_uniform_flux", "_sweep"]
    missing = [name for name in required if not hasattr(sampler, name)]
    if missing:
        raise AttributeError(f"sampler missing required function(s): {', '.join(missing)}")


def validate_links(U: np.ndarray, L: int, N: int) -> np.ndarray:
    arr = np.asarray(U)
    expected_shape = (L, L, L, 3)
    print(f"[Gate4pp Polyakov adapter] U.shape={arr.shape} dtype={arr.dtype} min={arr.min()} max={arr.max()}", flush=True)
    if arr.shape != expected_shape:
        raise ValueError(f"expected U.shape == {expected_shape}; got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"expected integer U dtype; got {arr.dtype}")
    if arr.min() < 0 or arr.max() >= N:
        raise ValueError(f"expected U values in {{0,...,{N - 1}}}; got min={arr.min()} max={arr.max()}")
    return arr


def initialize_links(sampler: ModuleType, L: int, N: int, start_mode: str, max_hot_tries: int) -> np.ndarray:
    if start_mode == "small_flux":
        return sampler._initialize_uniform_flux(L, N, 1)
    if start_mode == "large_flux":
        return sampler._initialize_uniform_flux(L, N, N // 2)
    return sampler._initialize_hot_no_flat(L, N, max_hot_tries)


def output_path(out_dir: Path, N: int, L: int, beta: float) -> Path:
    return out_dir / f"gate4pp_polyakov_N{N}_L{L}_{beta_file_label(beta)}.json"


def run_one_beta(
    sampler: ModuleType,
    N: int,
    L: int,
    beta: float,
    therm: int,
    sweeps: int,
    measure_every: int,
    seed: int,
    small_steps: bool,
    start_mode: str,
    max_hot_tries: int,
    out_dir: Path,
) -> Dict[str, Any]:
    if sweeps <= 0 or measure_every <= 0:
        raise ValueError("sweeps and measure_every must be positive")
    if sweeps % measure_every != 0:
        raise ValueError("sweeps must be divisible by measure_every")

    np.random.seed(seed)
    U = initialize_links(sampler, L=L, N=N, start_mode=start_mode, max_hot_tries=max_hot_tries)
    U = validate_links(U, L=L, N=N)

    print(
        f"[Gate4pp Polyakov adapter] beta={beta} N={N} L={L} seed={seed} start={start_mode} therm={therm} sweeps={sweeps} measure_every={measure_every}",
        flush=True,
    )

    t0 = time.time()
    for _ in range(therm):
        sampler._sweep(U, N, float(beta), bool(small_steps))

    acc = EnsemblePolyakovAccumulator(N=N, L=L, r_values=list(range(1, (L // 2) + 1)))
    accepted = 0
    attempted = 0
    total_links = L * L * L * 3
    n_cfg = 0

    for sweep in range(sweeps):
        accepted += int(sampler._sweep(U, N, float(beta), bool(small_steps)))
        attempted += total_links
        if (sweep + 1) % measure_every == 0:
            if n_cfg == 0:
                validate_links(U, L=L, N=N)
            acc.accumulate(U, label=f"beta={beta:.6g}:sweep={sweep+1}")
            n_cfg += 1

    packet = analyze_ensemble(acc)
    packet.update(
        {
            "beta": float(beta),
            "adapter": {
                "sampler_api": "_initialize_* + _sweep live-U adapter",
                "observable": "center-projected winding-line correlator",
                "start_mode": start_mode,
                "therm": therm,
                "sweeps": sweeps,
                "measure_every": measure_every,
                "seed": seed,
                "small_steps": bool(small_steps),
                "acceptance": accepted / attempted if attempted else float("nan"),
                "elapsed_sec": time.time() - t0,
            },
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_path(out_dir, N=N, L=L, beta=beta)
    packet["output_path"] = str(out_path)
    out_path.write_text(json.dumps(make_json_safe(packet), indent=2), encoding="utf-8")

    print(
        f"[Gate4pp Polyakov adapter] wrote {out_path} n_cfg={n_cfg} acceptance={packet['adapter']['acceptance']:.4f} elapsed={packet['adapter']['elapsed_sec']:.1f}s",
        flush=True,
    )
    print(f"[Gate4pp Polyakov adapter] classification: {packet.get('classification')}", flush=True)
    return packet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gate 4pp live-U adapter for center-projected winding-line correlators.")
    p.add_argument("--sampler", type=Path, required=True, help="Path to local b1b_zn3d_scaling_sampler.py")
    p.add_argument("--N", type=int, default=9, help="Z_N group order; Gate 4pp uses 9")
    p.add_argument("--L", type=int, default=12, help="Lattice side; first Gate 4pp run uses 12")
    p.add_argument("--betas", type=float, nargs="+", required=True, help="Beta values to run")
    p.add_argument("--therm", type=int, default=500, help="Thermalization sweeps")
    p.add_argument("--sweeps", type=int, default=1000, help="Measurement sweeps after thermalization")
    p.add_argument("--measure-every", type=int, default=10, help="Accumulate one config every k sweeps")
    p.add_argument("--seed", type=int, default=24680, help="Base RNG seed")
    p.add_argument("--small-steps", action="store_true", help="Use sampler +/-1 proposals")
    p.add_argument("--start-mode", choices=["hot", "small_flux", "large_flux"], default="hot")
    p.add_argument("--max-hot-tries", type=int, default=100, help="Hot no-flat initialization attempts")
    p.add_argument("--out-dir", type=Path, default=Path("gate4pp_polyakov_outputs"), help="Output directory for JSON packets")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sampler = load_sampler_module(args.sampler)
    require_sampler_api(sampler)

    for i, beta in enumerate(args.betas):
        seed = args.seed + int(round(float(beta) * 1000)) + 100000 * i
        run_one_beta(
            sampler=sampler,
            N=args.N,
            L=args.L,
            beta=float(beta),
            therm=args.therm,
            sweeps=args.sweeps,
            measure_every=args.measure_every,
            seed=seed,
            small_steps=args.small_steps,
            start_mode=args.start_mode,
            max_hot_tries=args.max_hot_tries,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()

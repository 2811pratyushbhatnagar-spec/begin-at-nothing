"""
Validate and summarize a Gate 4pp four-beta diagnostic packet grid.

This script is intentionally QC-only. It does not classify Branch C/D physics.
It checks that each expected diagnostic JSON packet has the healthy structure
required before any physics inspection.

Typical use after the first N=9, L=12, n_cfg=100 diagnostic grid:

    python research/gate4pp/validate_gate4pp_grid.py gate4pp_outputs \
        --expect-n-cfg 100 \
        --betas 1.2 1.9 2.13 2.8

Expected file names are produced by gate4pp_runner.py:

    gate4pp_N9_L12_beta_1p2000.json
    gate4pp_N9_L12_beta_1p9000.json
    gate4pp_N9_L12_beta_2p1300.json
    gate4pp_N9_L12_beta_2p8000.json

Protected boundary:

Gate 4pp Wilson-loop packets are code-level observables until uncertainty,
autocorrelation, finite-size, and plateau checks exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

try:  # package-style import
    from .validate_gate4pp_packet import validate_packet
except ImportError:  # script-style import from this directory
    from validate_gate4pp_packet import validate_packet


def beta_file_label(beta: float) -> str:
    """Match gate4pp_runner.py beta filename formatting."""
    return f"beta_{beta:.4f}".replace(".", "p")


def expected_packet_path(output_dir: Path, N: int, L: int, beta: float) -> Path:
    return output_dir / f"gate4pp_N{N}_L{L}_{beta_file_label(beta)}.json"


def _as_float(value: Any) -> float | None:
    try:
        if isinstance(value, dict) and "real" in value and "imag" in value:
            if abs(float(value.get("imag", 0.0))) > 1e-12:
                return None
            return float(value["real"])
        return float(value)
    except Exception:
        return None


def _find_fit_value(fit: Dict[str, Any], key: str) -> Any:
    if key in fit:
        return fit[key]
    for nested_key in ("W1", "W2", "combined", "fit"):
        nested = fit.get(nested_key)
        if isinstance(nested, dict) and key in nested:
            return nested[key]
    return None


def summarize_packet(beta: float, path: Path, packet: Dict[str, Any], ok: bool, errors: Sequence[str]) -> Dict[str, Any]:
    fit = packet.get("fit") if isinstance(packet.get("fit"), dict) else {}
    sigma = _as_float(_find_fit_value(fit, "sigma")) if isinstance(fit, dict) else None
    mu_perim = _as_float(_find_fit_value(fit, "mu_perim")) if isinstance(fit, dict) else None
    loop_results = packet.get("loop_results") if isinstance(packet.get("loop_results"), list) else []
    creutz = packet.get("creutz_ratios")
    creutz_count = len(creutz) if isinstance(creutz, (list, dict)) else None

    return {
        "beta": beta,
        "path": str(path),
        "ok": ok,
        "n_cfg": packet.get("n_cfg"),
        "loop_results": len(loop_results),
        "creutz_ratios": creutz_count,
        "sigma": sigma,
        "mu_perim": mu_perim,
        "classification": packet.get("classification"),
        "errors": "; ".join(errors),
    }


def load_packet(path: Path) -> Tuple[Dict[str, Any], List[str]]:
    if not path.exists():
        return {}, [f"missing packet: {path}"]
    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except Exception as exc:
        return {}, [f"could not read JSON packet {path}: {exc}"]


def validate_grid(output_dir: Path, betas: Sequence[float], N: int, L: int, expect_n_cfg: int | None) -> Tuple[bool, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    all_ok = True

    for beta in betas:
        path = expected_packet_path(output_dir, N=N, L=L, beta=beta)
        packet, load_errors = load_packet(path)
        if load_errors:
            rows.append(summarize_packet(beta, path, {}, False, load_errors))
            all_ok = False
            continue

        ok, errors = validate_packet(packet, expect_n_cfg=expect_n_cfg)
        rows.append(summarize_packet(beta, path, packet, ok, errors))
        if not ok:
            all_ok = False

    return all_ok, rows


def print_table(rows: Sequence[Dict[str, Any]]) -> None:
    headers = ["beta", "ok", "n_cfg", "loop_results", "creutz_ratios", "sigma", "mu_perim", "errors"]
    print("\t".join(headers))
    for row in rows:
        values = []
        for key in headers:
            value = row.get(key)
            if isinstance(value, float):
                values.append(f"{value:.8g}")
            else:
                values.append("" if value is None else str(value))
        print("\t".join(values))


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["beta", "path", "ok", "n_cfg", "loop_results", "creutz_ratios", "sigma", "mu_perim", "classification", "errors"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Gate 4pp diagnostic beta grid.")
    parser.add_argument("output_dir", type=Path, help="Directory containing Gate 4pp JSON packets")
    parser.add_argument("--expect-n-cfg", type=int, default=None, help="Expected accumulated configurations per beta")
    parser.add_argument("--N", type=int, default=9, help="Group order used in packet filenames")
    parser.add_argument("--L", type=int, default=12, help="Lattice side used in packet filenames")
    parser.add_argument("--betas", type=float, nargs="+", default=[1.2, 1.9, 2.13, 2.8], help="Expected beta values")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV summary output path")
    args = parser.parse_args()

    ok, rows = validate_grid(args.output_dir, betas=args.betas, N=args.N, L=args.L, expect_n_cfg=args.expect_n_cfg)
    print_table(rows)

    if args.csv is not None:
        write_csv(rows, args.csv)
        print(f"wrote CSV summary: {args.csv}")

    if ok:
        print("PASS: Gate 4pp diagnostic grid is structurally healthy and diagnostic-only")
        return 0

    print("FAIL: Gate 4pp diagnostic grid is not ready for physics inspection")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

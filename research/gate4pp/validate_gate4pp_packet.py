"""
Validate Gate 4pp diagnostic JSON packets before physics interpretation.

This script checks the first real sampler-to-runner handoff packet. It does not
classify Branch C/D physics. It only verifies that the JSON packet has the
minimum healthy structure expected from gate4pp_runner.py / analyze_ensemble().

Usage:

    python research/gate4pp/validate_gate4pp_packet.py path/to/packet.json
    python research/gate4pp/validate_gate4pp_packet.py path/to/packet.json --expect-n-cfg 10

Healthy first packet conditions:

- n_cfg exists and optionally matches --expect-n-cfg
- loop_results is a non-empty list
- each loop result has R, T, W1_abs, W2_abs
- fit exists and contains finite sigma and mu_perim values
- creutz_ratios exists as a list or dict; it may be empty depending on loop sizes
- classification exists and says diagnostic only
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REQUIRED_LOOP_KEYS = ("R", "T", "W1_abs", "W2_abs")


def _as_float(value: Any) -> float:
    if isinstance(value, dict) and "real" in value and "imag" in value:
        if abs(float(value.get("imag", 0.0))) > 1e-12:
            raise ValueError(f"expected real value; got complex-like {value!r}")
        return float(value["real"])
    return float(value)


def _finite_number(value: Any) -> bool:
    try:
        x = _as_float(value)
    except Exception:
        return False
    return math.isfinite(x)


def _find_fit_value(fit: Dict[str, Any], key: str) -> Any:
    """Find a fit scalar in either flat or mildly nested fit packets."""
    if key in fit:
        return fit[key]
    for nested_key in ("W1", "W2", "combined", "fit"):
        nested = fit.get(nested_key)
        if isinstance(nested, dict) and key in nested:
            return nested[key]
    raise KeyError(key)


def validate_packet(packet: Dict[str, Any], expect_n_cfg: int | None = None) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    n_cfg = packet.get("n_cfg")
    if not isinstance(n_cfg, int):
        errors.append("n_cfg missing or not an integer")
    elif expect_n_cfg is not None and n_cfg != expect_n_cfg:
        errors.append(f"n_cfg mismatch: expected {expect_n_cfg}, got {n_cfg}")

    loop_results = packet.get("loop_results")
    if not isinstance(loop_results, list) or not loop_results:
        errors.append("loop_results missing, not a list, or empty")
    else:
        for idx, row in enumerate(loop_results):
            if not isinstance(row, dict):
                errors.append(f"loop_results[{idx}] is not an object")
                continue
            for key in REQUIRED_LOOP_KEYS:
                if key not in row:
                    errors.append(f"loop_results[{idx}] missing key {key!r}")
            for key in ("W1_abs", "W2_abs"):
                if key in row and not _finite_number(row[key]):
                    errors.append(f"loop_results[{idx}].{key} is not finite")

    fit = packet.get("fit")
    if not isinstance(fit, dict):
        errors.append("fit missing or not an object")
    else:
        for key in ("sigma", "mu_perim"):
            try:
                value = _find_fit_value(fit, key)
            except KeyError:
                errors.append(f"fit missing {key!r}")
                continue
            if value is None or not _finite_number(value):
                errors.append(f"fit.{key} is missing or not finite")

    creutz = packet.get("creutz_ratios")
    if creutz is None:
        errors.append("creutz_ratios missing")
    elif not isinstance(creutz, (list, dict)):
        errors.append("creutz_ratios must be a list or object; it may be empty")

    classification = packet.get("classification")
    if not isinstance(classification, str) or not classification.strip():
        errors.append("classification missing or not a string")
    elif "diagnostic only" not in classification.lower():
        errors.append("classification does not say 'diagnostic only'")

    return (len(errors) == 0), errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Gate 4pp diagnostic JSON packet.")
    parser.add_argument("packet", type=Path, help="Path to Gate 4pp JSON packet")
    parser.add_argument("--expect-n-cfg", type=int, default=None, help="Expected number of accumulated configurations")
    args = parser.parse_args()

    try:
        packet = json.loads(args.packet.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"FAIL: could not read JSON packet: {exc}")
        return 2

    ok, errors = validate_packet(packet, expect_n_cfg=args.expect_n_cfg)
    if ok:
        print("PASS: Gate 4pp packet is structurally healthy and diagnostic-only")
        print(f"n_cfg: {packet.get('n_cfg')}")
        print(f"loop_results: {len(packet.get('loop_results', []))}")
        creutz = packet.get("creutz_ratios")
        if isinstance(creutz, list):
            print(f"creutz_ratios: {len(creutz)}")
        elif isinstance(creutz, dict):
            print(f"creutz_ratios: {len(creutz)}")
        else:
            print("creutz_ratios: unknown")
        print(f"classification: {packet.get('classification')}")
        return 0

    print("FAIL: Gate 4pp packet is not ready for physics inspection")
    for err in errors:
        print(f"- {err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

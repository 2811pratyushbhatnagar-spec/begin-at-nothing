"""
Validate Gate 4pp center-projected winding-line correlator packets.

This validator is handoff/QC-only. It does not classify confinement, Branch C/D,
string tension, or plateau behavior.

Usage:

    python research/gate4pp/validate_gate4pp_polyakov_packet.py \
      gate4pp_polyakov_outputs/gate4pp_polyakov_N9_L12_beta_2p8000.json \
      --expect-n-cfg 100
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

EXPECTED_R_VALUES = [1, 2, 3, 4, 5, 6]


def _as_float(value: Any) -> float:
    if isinstance(value, dict) and "real" in value and "imag" in value:
        # Complex average exists; magnitude/fit validators use real scalar fields.
        return float(value["real"])
    return float(value)


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(_as_float(value))
    except Exception:
        return False


def _is_complex_json(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if "real" not in value or "imag" not in value:
        return False
    return _finite_number(value["real"]) and _finite_number(value["imag"])


def _fit_finite_or_validly_insufficient(fit: Any, name: str, errors: List[str]) -> None:
    if not isinstance(fit, dict):
        errors.append(f"{name} missing or not an object")
        return
    mass = fit.get("mass")
    if mass is None:
        if fit.get("msg") != "insufficient signal above threshold":
            errors.append(f"{name}.mass is None without valid insufficient-signal message")
        return
    if not _finite_number(mass):
        errors.append(f"{name}.mass is not finite")
    c = fit.get("c")
    if c is not None and not _finite_number(c):
        errors.append(f"{name}.c is not finite")


def validate_packet(packet: Dict[str, Any], expect_n_cfg: int | None = None) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    if packet.get("N") != 9:
        errors.append(f"N mismatch: expected 9, got {packet.get('N')}")
    if packet.get("L") != 12:
        errors.append(f"L mismatch: expected 12, got {packet.get('L')}")

    n_cfg = packet.get("n_cfg")
    if not isinstance(n_cfg, int):
        errors.append("n_cfg missing or not an integer")
    elif expect_n_cfg is not None and n_cfg != expect_n_cfg:
        errors.append(f"n_cfg mismatch: expected {expect_n_cfg}, got {n_cfg}")

    r_values = packet.get("r_values")
    if r_values != EXPECTED_R_VALUES:
        errors.append(f"r_values mismatch: expected {EXPECTED_R_VALUES}, got {r_values}")

    results = packet.get("correlator_results")
    if not isinstance(results, list) or not results:
        errors.append("correlator_results missing, not a list, or empty")
    else:
        seen_r = []
        for idx, row in enumerate(results):
            if not isinstance(row, dict):
                errors.append(f"correlator_results[{idx}] is not an object")
                continue
            seen_r.append(row.get("r"))
            for key in ("r", "C1", "C2", "C1_abs", "C2_abs"):
                if key not in row:
                    errors.append(f"correlator_results[{idx}] missing key {key!r}")
            for key in ("C1", "C2"):
                if key in row and not _is_complex_json(row[key]):
                    errors.append(f"correlator_results[{idx}].{key} is not complex JSON with finite real/imag")
            for key in ("C1_abs", "C2_abs"):
                if key in row and not _finite_number(row[key]):
                    errors.append(f"correlator_results[{idx}].{key} is not finite")
        if seen_r != EXPECTED_R_VALUES:
            errors.append(f"correlator result r sequence mismatch: expected {EXPECTED_R_VALUES}, got {seen_r}")

    _fit_finite_or_validly_insufficient(packet.get("fit_C1"), "fit_C1", errors)
    _fit_finite_or_validly_insufficient(packet.get("fit_C2"), "fit_C2", errors)

    classification = packet.get("classification")
    if not isinstance(classification, str) or not classification.strip():
        errors.append("classification missing or not a string")
    elif "diagnostic only" not in classification.lower():
        errors.append("classification does not say 'diagnostic only'")

    return (len(errors) == 0), errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Gate 4pp winding-line correlator packet.")
    parser.add_argument("packet", type=Path)
    parser.add_argument("--expect-n-cfg", type=int, default=None)
    args = parser.parse_args()

    try:
        packet = json.loads(args.packet.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"FAIL: could not read JSON packet: {exc}")
        return 2

    ok, errors = validate_packet(packet, expect_n_cfg=args.expect_n_cfg)
    if ok:
        print("PASS: Gate 4pp winding-line packet is structurally healthy and diagnostic-only")
        print(f"N: {packet.get('N')}")
        print(f"L: {packet.get('L')}")
        print(f"n_cfg: {packet.get('n_cfg')}")
        print(f"r_values: {packet.get('r_values')}")
        print(f"classification: {packet.get('classification')}")
        return 0

    print("FAIL: Gate 4pp winding-line packet is not ready for physics inspection")
    for err in errors:
        print(f"- {err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""
Gate 4pp standalone runner for Z_N center-projected Wilson-loop measurement.

This file intentionally does not depend on the internal structure of the B1b /
Z_N sampler. It only requires access to link arrays with shape:

    U.shape == (L, L, L, 3)
    U[x, y, z, mu] in {0, ..., N-1}

The runner supports two integration styles:

1. Manual three-line integration inside an existing sampler loop:

    from gate4pp_runner import Gate4ppRunner

    runner = Gate4ppRunner(N=9, L=12, beta=beta)
    runner.accumulate(U, label="sweep_100")
    packet = runner.finalize()

2. Callable/object wrapper:

    packet = run_gate4pp(sampler, N=9, L=12)

where sampler is either:
    - a callable returning U, or
    - an object with .get_links() -> U

Register boundary
-----------------
This runner produces code-level observables and diagnostic analysis packets.
It does not classify Branch C/D physics. Physics classification requires ensemble
errors, autocorrelation checks, finite-size dependence, and visible Creutz plateau
behavior beyond this runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import json
import numpy as np

try:  # package-style import
    from .z9_center_wilson_loops import EnsembleWilsonAccumulator, analyze_ensemble, default_loop_sizes
except ImportError:  # script-style import from this directory
    from z9_center_wilson_loops import EnsembleWilsonAccumulator, analyze_ensemble, default_loop_sizes

LoopSize = Tuple[int, int]
LinkProvider = Union[Callable[[], np.ndarray], Any]


DEFAULT_BETA_GRID: Tuple[float, ...] = (1.2, 1.9, 2.13, 2.8)
FULL_BETA_GRID: Tuple[float, ...] = (0.8, 1.2, 1.6, 1.9, 2.13, 2.4, 2.8, 3.2)


@dataclass
class Gate4ppRunner:
    """Accumulate Gate 4pp Wilson-loop observables for one beta value."""

    N: int = 9
    L: int = 12
    beta: Optional[float] = None
    loop_sizes: Optional[Sequence[LoopSize]] = None
    output_dir: Optional[Union[str, Path]] = None
    labels: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        sizes = list(self.loop_sizes) if self.loop_sizes is not None else default_loop_sizes(self.L)
        self.loop_sizes = sizes
        self.accumulator = EnsembleWilsonAccumulator(N=self.N, L=self.L, loop_sizes=sizes)

    def accumulate(self, U: np.ndarray, label: str = "") -> None:
        """Add one measured configuration to the ensemble accumulator."""
        U_checked = self._validate_links(U)
        self.accumulator.add_configuration(U_checked, label=label)
        self.labels.append(label)

    def finalize(self, write_json: bool = False, filename: Optional[str] = None) -> Dict[str, Any]:
        """Analyze the ensemble and optionally write a JSON packet."""
        packet = analyze_ensemble(self.accumulator)
        packet.update(
            {
                "beta": self.beta,
                "register": "code-level observable; diagnostic-only physics status",
                "labels": list(self.labels),
                "loop_sizes": list(self.loop_sizes or []),
            }
        )

        if write_json:
            out_path = self._output_path(filename)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(make_json_safe(packet), indent=2), encoding="utf-8")
            packet["output_path"] = str(out_path)

        return packet

    def _validate_links(self, U: np.ndarray) -> np.ndarray:
        arr = np.asarray(U)
        expected_shape = (self.L, self.L, self.L, 3)
        if arr.shape != expected_shape:
            raise ValueError(f"Expected link array shape {expected_shape}; got {arr.shape}")
        if not np.issubdtype(arr.dtype, np.integer):
            raise TypeError("Gate 4pp links must be integer-valued")
        if np.any(arr < 0) or np.any(arr >= self.N):
            raise ValueError(f"Gate 4pp links must be in {{0,...,{self.N - 1}}}")
        return arr

    def _output_path(self, filename: Optional[str]) -> Path:
        out_dir = Path(self.output_dir) if self.output_dir is not None else Path("gate4pp_outputs")
        if filename is not None:
            return out_dir / filename
        beta_label = "beta_unknown" if self.beta is None else f"beta_{self.beta:.4f}".replace(".", "p")
        return out_dir / f"gate4pp_N{self.N}_L{self.L}_{beta_label}.json"


def get_links_from_sampler(sampler: LinkProvider) -> np.ndarray:
    """Extract a link array from a callable or object with .get_links()."""
    if callable(sampler):
        return np.asarray(sampler())
    if hasattr(sampler, "get_links") and callable(sampler.get_links):
        return np.asarray(sampler.get_links())
    raise TypeError("sampler must be callable or expose .get_links() -> np.ndarray")


def run_gate4pp(
    sampler: LinkProvider,
    N: int = 9,
    L: int = 12,
    beta_grid: Optional[Sequence[float]] = None,
    n_cfg: int = 1,
    update: Optional[Callable[[float, int], None]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    write_json: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Run Gate 4pp measurement over a beta grid using a minimal sampler hook.

    Parameters
    ----------
    sampler:
        Callable returning the current link array, or object with .get_links().
    N, L:
        Group order and lattice side.
    beta_grid:
        Beta values to label/drive measurement. Defaults to the limited Gate 4pp grid.
    n_cfg:
        Number of configurations to accumulate per beta. If no update function is
        supplied, this repeatedly samples the current state and is mainly useful
        for smoke/integration testing.
    update:
        Optional callback update(beta, cfg_index). The host sampler can use this
        to thermalize/advance between measurements. The runner does not assume
        sampler internals.
    output_dir, write_json:
        Optional output writing.

    Returns
    -------
    Mapping beta-label -> diagnostic analysis packet.
    """
    grid = tuple(beta_grid) if beta_grid is not None else DEFAULT_BETA_GRID
    packets: Dict[str, Dict[str, Any]] = {}

    for beta in grid:
        runner = Gate4ppRunner(N=N, L=L, beta=beta, output_dir=output_dir)
        for cfg_index in range(n_cfg):
            if update is not None:
                update(float(beta), cfg_index)
            U = get_links_from_sampler(sampler)
            runner.accumulate(U, label=f"beta={beta:.6g}:cfg={cfg_index}")

        beta_key = f"beta_{float(beta):.6g}"
        packets[beta_key] = runner.finalize(write_json=write_json)

    return packets


def make_json_safe(value: Any) -> Any:
    """Convert numpy/complex objects in analysis packets to JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value


def smoke_test_runner() -> Dict[str, Any]:
    """Smoke-test the runner without any external sampler."""
    N = 9
    L = 6
    flat = np.zeros((L, L, L, 3), dtype=int)
    runner = Gate4ppRunner(N=N, L=L, beta=2.13, loop_sizes=[(1, 1), (2, 2)])
    runner.accumulate(flat, label="flat-0")
    runner.accumulate(flat, label="flat-1")
    packet = runner.finalize(write_json=False)
    all_unit = all(abs(float(r["W1_abs"]) - 1.0) < 1e-12 for r in packet["loop_results"])
    return {
        "n_cfg": packet["n_cfg"],
        "all_W1_abs_unit_for_flat": bool(all_unit),
        "classification": packet["classification"],
        "all_pass": bool(packet["n_cfg"] == 2 and all_unit and "diagnostic only" in packet["classification"]),
    }


if __name__ == "__main__":
    checks = smoke_test_runner()
    for key, value in checks.items():
        print(f"{key}: {value}")
    if not checks["all_pass"]:
        raise SystemExit(1)

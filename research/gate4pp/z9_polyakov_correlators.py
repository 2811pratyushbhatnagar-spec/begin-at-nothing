"""
Gate 4pp - Z_N center-projected winding-line correlators.

This module defines a cheaper center-projected transport observable for the
Gate 4pp research track.

It should be called a Polyakov-style / winding-line correlator, not a full
finite-temperature Polyakov loop unless one lattice direction has explicitly
been declared temporal.

Core definitions
----------------
Given integer link variables:

    U[x, y, z, mu] in {0, ..., N-1}, mu in {0,1,2}

For a winding line in direction mu through a transverse base point x_perp:

    Phi_N_mu(x_perp) = sum_{s=0}^{L-1} U[x + s e_mu, mu] mod N
    Phi_3_mu(x_perp) = Phi_N_mu(x_perp) mod 3
    P_k_mu(x_perp) = exp(2*pi*i*k*Phi_3_mu(x_perp)/3)

Then for k=1,2:

    C_k(r) = < P_k_mu(x_perp) * conj(P_k_mu(x_perp + r e_nu)) >

averaged over mu, nu != mu, transverse base positions, and sampled
configurations.

Register boundary
-----------------
This module produces code-level diagnostic transport observables. It does not
classify confinement, Branch C/D, string tension, or continuum transport.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

Result = Dict[str, object]


def default_r_values(L: int) -> List[int]:
    """Gate 4pp default separation grid."""
    return list(range(1, (L // 2) + 1))


@dataclass
class PolyakovStyleMeasurer:
    """Measure center-projected winding-line correlators on one configuration."""

    U: np.ndarray
    N: int
    L: int
    N_center: int = 3

    def __post_init__(self) -> None:
        self.U = np.asarray(self.U)
        expected_shape = (self.L, self.L, self.L, 3)
        if self.U.shape != expected_shape:
            raise ValueError(f"U must have shape {expected_shape}; got {self.U.shape}")
        if self.N % self.N_center != 0:
            raise ValueError("N must be divisible by N_center")
        if not np.issubdtype(self.U.dtype, np.integer):
            raise TypeError("U must contain integer link variables")
        if np.any(self.U < 0) or np.any(self.U >= self.N):
            raise ValueError("U entries out of range")
        self.omega = np.exp(2j * np.pi / self.N_center)

    def _coord(self, site: np.ndarray) -> Tuple[int, int, int]:
        return tuple((site % self.L).astype(int))  # type: ignore[return-value]

    def winding_flux_modN(self, base: Tuple[int, int, int], mu: int) -> int:
        """Compute Phi_N for a winding line in direction mu."""
        if mu not in (0, 1, 2):
            raise ValueError("mu must be in {0,1,2}")
        s = np.array(base, dtype=int)
        flux = 0
        for _ in range(self.L):
            flux += int(self.U[self._coord(s)][mu])
            s[mu] = (s[mu] + 1) % self.L
        return int(flux % self.N)

    def winding_phase(self, base: Tuple[int, int, int], mu: int, k: int) -> complex:
        """Compute P_k_mu(base)."""
        if k not in (1, 2):
            raise ValueError("k must be 1 or 2")
        phi3 = self.winding_flux_modN(base, mu) % self.N_center
        return complex(self.omega ** (k * phi3))

    def _base_site(self, mu: int, a: int, b: int) -> np.ndarray:
        """Build a site using only the two coordinates transverse to mu."""
        site = np.zeros(3, dtype=int)
        transverse = [d for d in (0, 1, 2) if d != mu]
        site[transverse[0]] = a
        site[transverse[1]] = b
        return site

    def measure(self, r: int) -> Result:
        """Measure C_k(r), k=1,2, on one configuration."""
        if r <= 0 or r > self.L // 2:
            raise ValueError(f"r must be in 1..{self.L // 2}; got {r}")

        accum = {1: 0.0 + 0.0j, 2: 0.0 + 0.0j}
        n_pairs = 0
        phi3_hist = {1: np.zeros(self.N_center, dtype=int), 2: np.zeros(self.N_center, dtype=int)}

        for mu in (0, 1, 2):
            transverse = [d for d in (0, 1, 2) if d != mu]
            for nu in transverse:
                for a, b in product(range(self.L), repeat=2):
                    base = self._base_site(mu, a, b)
                    shifted = base.copy()
                    shifted[nu] = (shifted[nu] + r) % self.L

                    phi0 = self.winding_flux_modN(tuple(base), mu) % self.N_center
                    phir = self.winding_flux_modN(tuple(shifted), mu) % self.N_center
                    for k in (1, 2):
                        p0 = complex(self.omega ** (k * phi0))
                        pr = complex(self.omega ** (k * phir))
                        accum[k] += p0 * np.conjugate(pr)
                        phi3_hist[k][phi0] += 1
                    n_pairs += 1

        C1 = accum[1] / n_pairs
        C2 = accum[2] / n_pairs
        return result_packet(r, C1, C2, n_pairs, phi3_hist[1], phi3_hist[2])

    def measure_all(self, r_values: Optional[Sequence[int]] = None) -> List[Result]:
        values = list(r_values) if r_values is not None else default_r_values(self.L)
        return [self.measure(r) for r in values]


@dataclass
class EnsemblePolyakovAccumulator:
    """Accumulate complex winding-line correlators over configurations."""

    N: int
    L: int
    r_values: Optional[Sequence[int]] = None
    N_center: int = 3
    n_cfg: int = 0
    sum_C1: Dict[int, complex] = field(default_factory=dict)
    sum_C2: Dict[int, complex] = field(default_factory=dict)
    sum_phi3_hist_k1: Dict[int, np.ndarray] = field(default_factory=dict)
    sum_phi3_hist_k2: Dict[int, np.ndarray] = field(default_factory=dict)
    n_pairs_per_cfg: Dict[int, int] = field(default_factory=dict)
    cfg_labels: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.r_values = list(self.r_values) if self.r_values is not None else default_r_values(self.L)
        for r in self.r_values:
            rr = int(r)
            self.sum_C1[rr] = 0j
            self.sum_C2[rr] = 0j
            self.sum_phi3_hist_k1[rr] = np.zeros(self.N_center, dtype=int)
            self.sum_phi3_hist_k2[rr] = np.zeros(self.N_center, dtype=int)

    def accumulate(self, U: np.ndarray, label: str = "") -> None:
        measurer = PolyakovStyleMeasurer(U=U, N=self.N, L=self.L, N_center=self.N_center)
        for res in measurer.measure_all(self.r_values):
            r = int(res["r"])
            self.sum_C1[r] += complex(res["C1"])
            self.sum_C2[r] += complex(res["C2"])
            self.sum_phi3_hist_k1[r] += np.asarray(res["phi3_hist_k1"], dtype=int)
            self.sum_phi3_hist_k2[r] += np.asarray(res["phi3_hist_k2"], dtype=int)
            self.n_pairs_per_cfg[r] = int(res["n_pairs"])
        self.n_cfg += 1
        self.cfg_labels.append(label)

    def average_results(self) -> List[Result]:
        if self.n_cfg <= 0:
            raise ValueError("No configurations have been accumulated")
        out = []
        for r in self.r_values or []:
            rr = int(r)
            C1 = self.sum_C1[rr] / self.n_cfg
            C2 = self.sum_C2[rr] / self.n_cfg
            out.append(
                result_packet(
                    rr,
                    C1,
                    C2,
                    self.n_pairs_per_cfg.get(rr, 0) * self.n_cfg,
                    self.sum_phi3_hist_k1[rr],
                    self.sum_phi3_hist_k2[rr],
                )
            )
        return out


def result_packet(r: int, C1: complex, C2: complex, n_pairs: int, phi3_hist_k1: np.ndarray, phi3_hist_k2: np.ndarray) -> Result:
    return {
        "r": int(r),
        "C1": complex(C1),
        "C2": complex(C2),
        "C1_abs": float(abs(C1)),
        "C2_abs": float(abs(C2)),
        "C1_arg": float(np.angle(C1)),
        "C2_arg": float(np.angle(C2)),
        "implementation_sym_check": float(abs(abs(C1) - abs(C2))),
        "n_pairs": int(n_pairs),
        "phi3_hist_k1": np.asarray(phi3_hist_k1, dtype=int),
        "phi3_hist_k2": np.asarray(phi3_hist_k2, dtype=int),
    }


def fit_decay(results: Sequence[Mapping[str, object]], key: str = "C1_abs", min_abs: float = 1e-8) -> Dict[str, object]:
    """Fit -log(abs C(r)) = mass*r + c over finite signal above min_abs."""
    rows = [(int(row["r"]), float(row[key])) for row in results if float(row.get(key, 0.0)) > min_abs]
    if len(rows) < 2:
        return {"mass": None, "c": None, "residuals": None, "n_fit": len(rows), "msg": "insufficient signal above threshold"}
    x = np.array([r for r, _ in rows], dtype=float)
    y = np.array([-np.log(abs_c) for _, abs_c in rows], dtype=float)
    A = np.array([[r, 1.0] for r in x], dtype=float)
    coeffs, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return {
        "mass": float(coeffs[0]),
        "c": float(coeffs[1]),
        "residuals": float(residuals[0]) if len(residuals) > 0 else None,
        "n_fit": len(rows),
        "min_abs": float(min_abs),
        "key": key,
    }


def phase_diagnostic(results: Sequence[Mapping[str, object]], min_abs: float = 1e-8, phase_tol: float = 0.05) -> Dict[str, object]:
    flags = []
    active = []
    for row in results:
        c_abs = float(row.get("C1_abs", 0.0))
        c_arg = float(row.get("C1_arg", 0.0))
        if c_abs > min_abs:
            active.append(abs(c_arg))
            if abs(c_arg) > phase_tol:
                flags.append((int(row["r"]), c_arg))
    return {"max_arg_C1": max(active, default=0.0), "flags": flags, "min_abs": min_abs, "phase_tol": phase_tol}


def classify_diagnostic_only(results: Sequence[Mapping[str, object]]) -> str:
    if not results:
        return "diagnostic only: no winding-line correlator results available"
    return (
        "diagnostic only: winding-line correlators computed, but confinement, "
        "Branch C/D, string tension, and plateau claims require uncertainty, "
        "autocorrelation, finite-size, and fit-stability checks"
    )


def analyze_ensemble(acc: EnsemblePolyakovAccumulator) -> Dict[str, object]:
    results = acc.average_results()
    return {
        "N": acc.N,
        "L": acc.L,
        "n_cfg": acc.n_cfg,
        "r_values": list(acc.r_values or []),
        "correlator_results": results,
        "fit_C1": fit_decay(results, key="C1_abs"),
        "fit_C2": fit_decay(results, key="C2_abs"),
        "phase_diagnostic": phase_diagnostic(results),
        "classification": classify_diagnostic_only(results),
        "register": "Polyakov-style / winding-line correlator; diagnostic-only physics status",
    }


def make_json_safe(value):
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


def smoke_tests() -> Dict[str, object]:
    N = 9
    L = 6
    r_values = [1, 2, 3]
    flat = np.zeros((L, L, L, 3), dtype=int)
    flat_results = PolyakovStyleMeasurer(flat, N, L).measure_all(r_values)
    flat_ok = all(np.isclose(float(row["C1_abs"]), 1.0) and np.isclose(float(row["C2_abs"]), 1.0) for row in flat_results)
    uniform = np.ones((L, L, L, 3), dtype=int)
    uniform_results = PolyakovStyleMeasurer(uniform, N, L).measure_all(r_values)
    uniform_ok = all(np.isclose(float(row["C1_abs"]), 1.0) and np.isclose(float(row["C2_abs"]), 1.0) for row in uniform_results)
    rng = np.random.default_rng(1234)
    random_U = rng.integers(0, N, size=(L, L, L, 3), dtype=int)
    random_results = PolyakovStyleMeasurer(random_U, N, L).measure_all(r_values)
    random_finite = all(np.isfinite(float(row["C1_abs"])) and np.isfinite(float(row["C2_abs"])) for row in random_results)
    acc = EnsemblePolyakovAccumulator(N=N, L=L, r_values=r_values)
    acc.accumulate(flat, label="flat")
    acc.accumulate(uniform, label="uniform")
    packet = analyze_ensemble(acc)
    ensemble_ok = packet["n_cfg"] == 2 and all(np.isclose(float(row["C1_abs"]), 1.0) for row in packet["correlator_results"])
    return {
        "flat_correlators_unit": bool(flat_ok),
        "uniform_correlators_unit": bool(uniform_ok),
        "random_correlators_finite": bool(random_finite),
        "multi_config_accumulator_averages_before_analysis": bool(ensemble_ok),
        "classification": packet["classification"],
        "all_pass": bool(flat_ok and uniform_ok and random_finite and ensemble_ok),
    }


if __name__ == "__main__":
    checks = smoke_tests()
    for key, value in checks.items():
        print(f"{key}: {value}")
    if not checks["all_pass"]:
        raise SystemExit(1)

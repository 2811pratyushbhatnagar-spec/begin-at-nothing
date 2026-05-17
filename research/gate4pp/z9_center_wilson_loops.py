"""
Gate 4pp - Z_N center-projected N-ality Wilson-loop measurement.

Purpose
-------
Run-safe observable module for the Gate 4pp research track.

It measures center-projected Wilson-loop-like observables on a 3D additive
Z_N link lattice, with the intended Gate 4pp case N=9 and Z_3 center
projection.

Core observable
---------------
Given integer link variables:

    U[x, y, z, mu] in {0, ..., N-1}, mu in {0,1,2}

for a rectangular closed loop C, compute:

    Phi_N(C) = oriented link sum mod N
    Phi_3(C) = Phi_N(C) mod 3
    W_k(R,T) = < exp(2*pi*i*k*Phi_3(C)/3) >, k=1,2

Important register boundary
---------------------------
Single-configuration measurements are diagnostic only. Real Wilson-loop
analysis must average complex W_k values over an ensemble first, then compute
magnitudes, phases, area/perimeter fits, and Creutz ratios.

Physics classification is intentionally diagnostic-only here. Branch C/D claims
require ensemble errors, autocorrelation checks, finite-size dependence, and
visible Creutz-plateau behavior outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

LoopSize = Tuple[int, int]
Plane = Tuple[int, int]
Result = Dict[str, object]


DEFAULT_PLANES: Tuple[Plane, ...] = ((0, 1), (0, 2), (1, 2))


def default_loop_sizes(L: int) -> List[LoopSize]:
    """Return the Gate 4pp default loop grid with R,T <= L//3."""
    max_loop = L // 3
    square = [(r, r) for r in range(1, min(6, max_loop + 1))]
    asym = [(r, t) for (r, t) in ((2, 4), (3, 6), (4, 8)) if r <= max_loop and t <= max_loop]
    return square + asym


@dataclass
class WilsonLoopMeasurer:
    """Measure center-projected loops on one 3D Z_N link configuration.

    Parameters
    ----------
    U:
        Integer link field with shape (L, L, L, 3).
    N:
        Additive group order. Gate 4pp uses N=9.
    L:
        Lattice side length.
    """

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
            raise ValueError(f"N={self.N} must be divisible by N_center={self.N_center}")
        if not np.issubdtype(self.U.dtype, np.integer):
            raise TypeError("U must contain integer link variables")
        if np.any(self.U < 0) or np.any(self.U >= self.N):
            raise ValueError(f"U entries must lie in {{0,...,{self.N - 1}}}")
        self.omega = np.exp(2j * np.pi / self.N_center)

    def _coord(self, site: np.ndarray) -> Tuple[int, int, int]:
        return tuple((site % self.L).astype(int))  # type: ignore[return-value]

    def loop_flux_modN(self, x0: int, y0: int, z0: int, mu: int, nu: int, R: int, T: int) -> int:
        """Compute Phi_N(C), the oriented rectangular loop flux mod N."""
        if mu == nu or mu not in (0, 1, 2) or nu not in (0, 1, 2):
            raise ValueError("mu and nu must be distinct directions in {0,1,2}")
        if R <= 0 or T <= 0:
            raise ValueError("R and T must be positive")

        site = np.array([x0, y0, z0], dtype=int)
        flux = 0

        # Forward mu-links: R steps.
        s = site.copy()
        for _ in range(R):
            flux += int(self.U[self._coord(s)][mu])
            s[mu] = (s[mu] + 1) % self.L

        # Forward nu-links: T steps from x0 + R*mu_hat.
        for _ in range(T):
            flux += int(self.U[self._coord(s)][nu])
            s[nu] = (s[nu] + 1) % self.L

        # Backward mu-links: subtract the link at the stepped-back site.
        for _ in range(R):
            s[mu] = (s[mu] - 1) % self.L
            flux -= int(self.U[self._coord(s)][mu])

        # Backward nu-links: subtract the link at the stepped-back site.
        for _ in range(T):
            s[nu] = (s[nu] - 1) % self.L
            flux -= int(self.U[self._coord(s)][nu])

        return int(flux % self.N)

    def loop_flux_mod3(self, x0: int, y0: int, z0: int, mu: int, nu: int, R: int, T: int) -> int:
        """Center projection: Phi_3 = Phi_N mod 3."""
        return self.loop_flux_modN(x0, y0, z0, mu, nu, R, T) % self.N_center

    def measure(self, R: int, T: int, planes: Optional[Sequence[Plane]] = None) -> Result:
        """Measure W1/W2 for a single configuration.

        This is diagnostic only. Use EnsembleWilsonAccumulator for physics-facing
        analysis.
        """
        active_planes = tuple(planes) if planes is not None else DEFAULT_PLANES
        accum_W1 = 0.0 + 0.0j
        accum_W2 = 0.0 + 0.0j
        phi3_hist = np.zeros(self.N_center, dtype=int)
        n_loops = 0

        for (mu, nu) in active_planes:
            if mu == nu:
                raise ValueError("plane directions must be distinct")
            for x0, y0, z0 in product(range(self.L), repeat=3):
                phi3 = self.loop_flux_mod3(x0, y0, z0, mu, nu, R, T)
                phi3_hist[phi3] += 1
                accum_W1 += self.omega ** phi3
                accum_W2 += self.omega ** (2 * phi3)
                n_loops += 1

        W1 = accum_W1 / n_loops
        W2 = accum_W2 / n_loops
        return result_packet(R=R, T=T, W1=W1, W2=W2, phi3_hist=phi3_hist, n_loops=n_loops)

    def measure_all(
        self,
        loop_sizes: Optional[Sequence[LoopSize]] = None,
        planes: Optional[Sequence[Plane]] = None,
    ) -> List[Result]:
        """Measure all requested loop sizes on one configuration."""
        sizes = list(loop_sizes) if loop_sizes is not None else default_loop_sizes(self.L)
        return [self.measure(R, T, planes=planes) for (R, T) in sizes]


@dataclass
class EnsembleWilsonAccumulator:
    """Accumulate complex Wilson-loop values over configurations before fitting."""

    N: int
    L: int
    loop_sizes: Optional[Sequence[LoopSize]] = None
    planes: Optional[Sequence[Plane]] = None
    N_center: int = 3
    n_cfg: int = 0
    sum_W1: Dict[LoopSize, complex] = field(default_factory=dict)
    sum_W2: Dict[LoopSize, complex] = field(default_factory=dict)
    sum_phi3_hist: Dict[LoopSize, np.ndarray] = field(default_factory=dict)
    n_loops_per_cfg: Dict[LoopSize, int] = field(default_factory=dict)
    cfg_labels: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.loop_sizes is None:
            self.loop_sizes = default_loop_sizes(self.L)
        else:
            self.loop_sizes = list(self.loop_sizes)
        for size in self.loop_sizes:
            self.sum_W1[size] = 0.0 + 0.0j
            self.sum_W2[size] = 0.0 + 0.0j
            self.sum_phi3_hist[size] = np.zeros(self.N_center, dtype=int)

    def add_configuration(self, U: np.ndarray, label: str = "") -> None:
        """Measure one configuration and add its complex W values to the ensemble sums."""
        measurer = WilsonLoopMeasurer(U=U, N=self.N, L=self.L, N_center=self.N_center)
        results = measurer.measure_all(loop_sizes=self.loop_sizes, planes=self.planes)

        for r in results:
            key = (int(r["R"]), int(r["T"]))
            self.sum_W1[key] += complex(r["W1"])
            self.sum_W2[key] += complex(r["W2"])
            self.sum_phi3_hist[key] += np.asarray(r["phi3_hist"], dtype=int)
            self.n_loops_per_cfg[key] = int(r["n_loops"])

        self.n_cfg += 1
        self.cfg_labels.append(label)

    def average_results(self) -> List[Result]:
        """Return ensemble-averaged W1/W2 packets. Requires n_cfg > 0."""
        if self.n_cfg <= 0:
            raise ValueError("No configurations have been accumulated")

        averaged: List[Result] = []
        for (R, T) in self.loop_sizes or []:
            W1_avg = self.sum_W1[(R, T)] / self.n_cfg
            W2_avg = self.sum_W2[(R, T)] / self.n_cfg
            averaged.append(
                result_packet(
                    R=R,
                    T=T,
                    W1=W1_avg,
                    W2=W2_avg,
                    phi3_hist=self.sum_phi3_hist[(R, T)],
                    n_loops=self.n_loops_per_cfg.get((R, T), 0) * self.n_cfg,
                )
            )
        return averaged


def result_packet(R: int, T: int, W1: complex, W2: complex, phi3_hist: np.ndarray, n_loops: int) -> Result:
    """Normalize the result schema and diagnostic fields."""
    return {
        "R": int(R),
        "T": int(T),
        "W1": complex(W1),
        "W2": complex(W2),
        "W1_abs": float(abs(W1)),
        "W2_abs": float(abs(W2)),
        "W1_arg": float(np.angle(W1)),
        "W2_arg": float(np.angle(W2)),
        "implementation_sym_check": float(abs(abs(W1) - abs(W2))),
        "phi3_hist": np.asarray(phi3_hist, dtype=int),
        "n_loops": int(n_loops),
    }


def creutz_ratio(results: Sequence[Mapping[str, object]]) -> List[Dict[str, float]]:
    """Compute Creutz ratios from ensemble-averaged W1 magnitudes."""
    wmap = {(int(r["R"]), int(r["T"])): float(r["W1_abs"]) for r in results}
    ratios: List[Dict[str, float]] = []

    for r in results:
        R, T = int(r["R"]), int(r["T"])
        if R < 2 or T < 2:
            continue
        needed = [(R, T), (R - 1, T - 1), (R, T - 1), (R - 1, T)]
        if not all(k in wmap for k in needed):
            continue

        w_RT = wmap[(R, T)]
        w_R1T1 = wmap[(R - 1, T - 1)]
        w_RT1 = wmap[(R, T - 1)]
        w_R1T = wmap[(R - 1, T)]

        denom = w_RT1 * w_R1T
        numer = w_RT * w_R1T1
        if denom < 1e-15 or numer < 1e-15:
            continue

        chi = -np.log(numer / denom)
        ratios.append({"R": float(R), "T": float(T), "chi1": float(chi)})

    return ratios


def fit_area_perimeter(results: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Fit -log|W1| = sigma*RT + mu*2(R+T) + c on ensemble averages."""
    rows = [(int(r["R"]), int(r["T"]), float(r["W1_abs"])) for r in results if float(r["W1_abs"]) > 1e-15]
    if len(rows) < 3:
        return {"sigma": None, "mu_perim": None, "c": None, "residuals": None, "msg": "insufficient data"}

    y = np.array([-np.log(w) for (_, _, w) in rows], dtype=float)
    A = np.array([[R * T, 2 * (R + T), 1.0] for (R, T, _) in rows], dtype=float)
    coeffs, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return {
        "sigma": float(coeffs[0]),
        "mu_perim": float(coeffs[1]),
        "c": float(coeffs[2]),
        "residuals": float(residuals[0]) if len(residuals) > 0 else None,
    }


def phase_diagnostic(results: Sequence[Mapping[str, object]], min_abs: float = 1e-8, phase_tol: float = 0.05) -> Dict[str, object]:
    """Flag phases only when the magnitude is large enough for arg(W1) to be meaningful."""
    active_args = [abs(float(r["W1_arg"])) for r in results if float(r["W1_abs"]) > min_abs]
    flags = [
        (int(r["R"]), int(r["T"]), float(r["W1_arg"]))
        for r in results
        if float(r["W1_abs"]) > min_abs and abs(float(r["W1_arg"])) > phase_tol
    ]
    return {"max_arg_W1": max(active_args, default=0.0), "flags": flags, "min_abs": min_abs, "phase_tol": phase_tol}


def classify_diagnostic_only(results: Sequence[Mapping[str, object]], ratios: Sequence[Mapping[str, float]]) -> str:
    """Return only a diagnostic status, never a Branch C/D physics classification."""
    if not results:
        return "diagnostic only: no Wilson-loop results available"
    if not ratios:
        return "diagnostic only: no Creutz ratios computed; no physics classification"
    return (
        "diagnostic only: Creutz ratios computed, but Branch C/D classification requires "
        "ensemble errors, autocorrelation checks, finite-size dependence, and visible plateau behavior"
    )


def analyze_ensemble(acc: EnsembleWilsonAccumulator) -> Dict[str, object]:
    """Analyze an accumulated ensemble. This is the only fitting path."""
    results = acc.average_results()
    ratios = creutz_ratio(results)
    fit = fit_area_perimeter(results)
    phase = phase_diagnostic(results)
    classification = classify_diagnostic_only(results, ratios)
    return {
        "N": acc.N,
        "L": acc.L,
        "n_cfg": acc.n_cfg,
        "loop_results": results,
        "creutz_ratios": ratios,
        "fit": fit,
        "phase_diagnostic": phase,
        "classification": classification,
    }


def measure_single_config_diagnostic(
    U: np.ndarray,
    N: int,
    L: int,
    beta: Optional[float] = None,
    loop_sizes: Optional[Sequence[LoopSize]] = None,
    label: str = "",
) -> Dict[str, object]:
    """Diagnostic single-config measurement. Does not produce physics classification."""
    measurer = WilsonLoopMeasurer(U=U, N=N, L=L)
    results = measurer.measure_all(loop_sizes=loop_sizes)
    ratios = creutz_ratio(results)
    fit = fit_area_perimeter(results)
    phase = phase_diagnostic(results)
    return {
        "N": N,
        "L": L,
        "beta": beta,
        "label": label,
        "register": "single configuration diagnostic only",
        "loop_results": results,
        "creutz_ratios": ratios,
        "fit": fit,
        "phase_diagnostic": phase,
        "classification": classify_diagnostic_only(results, ratios),
    }


def new_gate4pp_accumulator(N: int = 9, L: int = 12, loop_sizes: Optional[Sequence[LoopSize]] = None) -> EnsembleWilsonAccumulator:
    """Integration hook for sampler code."""
    return EnsembleWilsonAccumulator(N=N, L=L, loop_sizes=loop_sizes)


def smoke_tests() -> Dict[str, object]:
    """Run basic invariance and accumulator checks."""
    N = 9
    L = 6
    loop_sizes = [(1, 1), (1, 2), (2, 1), (2, 2)]

    flat = np.zeros((L, L, L, 3), dtype=int)
    flat_res = WilsonLoopMeasurer(flat, N, L).measure(1, 1)
    flat_ok = np.isclose(flat_res["W1_abs"], 1.0) and np.isclose(flat_res["W2_abs"], 1.0) and flat_res["phi3_hist"][0] == flat_res["n_loops"]

    uniform = np.ones((L, L, L, 3), dtype=int)
    uniform_res = WilsonLoopMeasurer(uniform, N, L).measure(2, 2)
    uniform_ok = np.isclose(uniform_res["W1_abs"], 1.0) and uniform_res["phi3_hist"][0] == uniform_res["n_loops"]

    rng = np.random.default_rng(1234)
    random_U = rng.integers(0, N, size=(L, L, L, 3), dtype=int)
    random_res = WilsonLoopMeasurer(random_U, N, L).measure(1, 1)
    hist = np.asarray(random_res["phi3_hist"], dtype=float)
    hist_frac = hist / hist.sum()
    random_hist_roughly_uniform = bool(np.all(np.abs(hist_frac - (1.0 / 3.0)) < 0.20))

    acc = EnsembleWilsonAccumulator(N=N, L=L, loop_sizes=loop_sizes)
    acc.add_configuration(flat, label="flat")
    acc.add_configuration(uniform, label="uniform")
    packet = analyze_ensemble(acc)
    ensemble_ok = acc.n_cfg == 2 and all(np.isclose(float(r["W1_abs"]), 1.0) for r in packet["loop_results"])

    return {
        "flat_config_W1_W2_equal_1": bool(flat_ok),
        "uniform_config_closed_loops_cancel": bool(uniform_ok),
        "random_config_phi3_hist_roughly_uniform": random_hist_roughly_uniform,
        "multi_config_accumulator_averages_before_analysis": bool(ensemble_ok),
        "random_phi3_hist_fraction": hist_frac.tolist(),
        "diagnostic_classification": packet["classification"],
        "all_pass": bool(flat_ok and uniform_ok and random_hist_roughly_uniform and ensemble_ok),
    }


if __name__ == "__main__":
    checks = smoke_tests()
    for key, value in checks.items():
        print(f"{key}: {value}")
    if not checks["all_pass"]:
        raise SystemExit(1)

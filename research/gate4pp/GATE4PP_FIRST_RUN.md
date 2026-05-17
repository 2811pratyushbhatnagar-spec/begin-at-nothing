# Gate 4pp — First Run Protocol v0.2

Status: research-track protocol

This protocol is for the first real sampler-to-runner handoff after the Gate 4pp observable, runner, packet validator, and grid validator have been added.

It does not classify physics.

It only checks whether the sampler can hand valid Z_N link configurations to the Gate 4pp Wilson-loop runner and produce structurally healthy diagnostic packets.

---

## 1. Required files

The Gate 4pp research folder should contain:

```text
research/gate4pp/z9_center_wilson_loops.py
research/gate4pp/gate4pp_runner.py
research/gate4pp/validate_gate4pp_packet.py
research/gate4pp/validate_gate4pp_grid.py
research/gate4pp/GATE4PP_FIRST_RUN.md
```

---

## 2. Required sampler interface

The runner does not require access to sampler internals.

It only needs a link array:

```python
U.shape == (L, L, L, 3)
U[x, y, z, mu] in {0, ..., N-1}
```

For the first Gate 4pp run:

```text
N = 9
L = 12
```

The host sampler can expose either:

```python
sampler.get_links() -> U
```

or:

```python
sampler() -> U
```

or directly call:

```python
runner.accumulate(U, label="...")
```

inside the measurement loop.

---

## 3. Minimal integration pattern

Inside the local sampler script:

```python
from research.gate4pp.gate4pp_runner import Gate4ppRunner

runner = Gate4ppRunner(N=9, L=12, beta=beta)

# after each measurement sweep:
runner.accumulate(U, label=f"sweep_{sweep}")

# after all measurements at this beta:
packet = runner.finalize(write_json=True)
```

This writes a diagnostic JSON packet under:

```text
gate4pp_outputs/
```

unless a custom output directory is provided.

---

## 4. First smoke packet

First run only one beta:

```text
N = 9
L = 12
beta = 2.13
n_cfg = 5 to 10
```

Recommended first packet:

```text
gate4pp_outputs/gate4pp_N9_L12_beta_2p1300.json
```

Validate it:

```bash
python research/gate4pp/validate_gate4pp_packet.py gate4pp_outputs/gate4pp_N9_L12_beta_2p1300.json --expect-n-cfg 10
```

If only 5 configurations were used:

```bash
python research/gate4pp/validate_gate4pp_packet.py gate4pp_outputs/gate4pp_N9_L12_beta_2p1300.json --expect-n-cfg 5
```

---

## 5. Healthy first packet criteria

The packet is healthy if:

- `n_cfg` matches the number of accumulated configurations.
- `loop_results` exists and is non-empty.
- each loop result contains `R`, `T`, `W1_abs`, and `W2_abs`.
- `fit.sigma` is finite.
- `fit.mu_perim` is finite.
- `creutz_ratios` exists; it may be empty depending on loop sizes.
- `classification` contains `diagnostic only`.

If this fails, do not inspect physics. Fix sampler handoff shape, dtype, output path, or packet schema first.

---

## 6. First diagnostic beta grid

Only after the beta=2.13 packet validates, run:

```text
N = 9
L = 12
beta_grid = [1.2, 1.9, 2.13, 2.8]
n_cfg = 100
```

This is still diagnostic.

The first grid asks whether the observable behaves coherently across the mode-locking region. It does not establish confinement, continuum transport, or Branch C/D classification.

---

## 7. Validate the four-beta grid

After the four beta packets are written, validate the grid structurally:

```bash
python research/gate4pp/validate_gate4pp_grid.py gate4pp_outputs \
  --expect-n-cfg 100 \
  --betas 1.2 1.9 2.13 2.8
```

Optional CSV summary:

```bash
python research/gate4pp/validate_gate4pp_grid.py gate4pp_outputs \
  --expect-n-cfg 100 \
  --betas 1.2 1.9 2.13 2.8 \
  --csv gate4pp_outputs/gate4pp_grid_summary.csv
```

A passing grid means only:

```text
All expected diagnostic packets are structurally healthy and remain diagnostic-only.
```

It does not mean the center-vortex branch is confirmed.

---

## 8. What not to claim after this run

Do not claim:

- confinement derived.
- center-vortex branch confirmed.
- continuum Yang-Mills transport established.
- Branch C or Branch D resolved.
- physical string tension measured.

The grid only produces first diagnostic Wilson-loop packets.

---

## 9. What can be claimed if validators pass

If the first packet validates:

```text
The sampler-to-runner handoff is structurally healthy.
```

If the four-beta grid completes and packets validate:

```text
Gate 4pp has first diagnostic N=9, L=12 Wilson-loop packets across the beta-locking region.
```

That is all.

---

## 10. Next gate after the diagnostic grid

The next gate is not interpretation. It is quality control:

- ensemble error estimates.
- autocorrelation checks.
- hot/cold or seed comparison.
- finite-size comparison.
- visible Creutz-plateau behavior.

Only after those exist can Branch C / Branch D language be reopened.

Protected line:

```text
Gate 4pp Wilson-loop packets are code-level observables until uncertainty, autocorrelation, finite-size, and plateau checks exist.
```

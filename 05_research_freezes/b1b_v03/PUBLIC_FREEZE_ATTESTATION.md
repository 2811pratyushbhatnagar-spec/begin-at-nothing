# Public Freeze Attestation — B1b Register Integrity Gate v0.3

Status: frozen before any v0.3 production chain.

This public record commits to the exact pre-registration and seed schedule for the beta=2.5 independent-confirmation gate.

## Frozen design

- L=12: 48 entirely new chains, balanced 24/24 by start family.
- L=16: 12 entirely new chains, balanced 6/6 by start family.
- Total: 60 entirely new chains.
- All v0.2.2 chains are excluded from every v0.3 confirmatory estimate and verdict.
- No adaptive stopping, top-up, or within-gate extension.
- A realized L=12 precision miss records v0.3 as FAIL; any continuation requires a separately audited and frozen v0.4.

## Frozen thresholds

- Endpoint chi(2,2) relative 95% half-width: <= 1.5%.
- Two-endpoint start-family gross-trapping tolerance: 5%, with 95% two-sided Bonferroni coverage.
- Relative L12-to-L16 drift: PASS iff q0.975 <= 5%; FAIL iff q0.025 > 5%; otherwise UNRESOLVED.
- Beta=2.5 drift may be called certified only if both endpoints pass every endpoint requirement.

## Cryptographic identifiers

PREREGISTRATION.json SHA256:

`fa403e4967211a25fa72ef5eca4e1313a2dcbc6073c8e24748aab376d8200e26`

Complete frozen archive SHA256:

`cb6622eb46a9808c175acce6435c4fc59a0906c03b135d0aa10fb30c1a0c4a90`

MANIFEST.csv SHA256:

`7710f7e408ade3070c9e3207139d57325b7fc69050b9347628427e830d028ef6`

## Timestamp boundary

This GitHub-hosted record is a public server-side publication trace. It closes the private file-drawer gap, but it is not treated as the sole permanent independent timestamp. The packet should additionally be anchored through Zenodo and/or OpenTimestamps.

No chain may enter the v0.3 confirmatory cohort unless its execution begins after this public record exists.

Any change to the frozen design requires a new version, new hashes, and a new public pre-execution record.

# Agent Runtime Rules

v0.1 does not provide runtime execution.

These rules are for future implementations.

## Minimum execution gate

An agent action may proceed only if all are true:

- Promise is authorized by the relevant human or lawful process.
- Boundary states what may be accessed, affected, revealed, stored, or changed.
- Cost budget is declared.
- Trace requirement is accepted.
- Repair path exists.
- Human review occurs when the action affects another person, money, identity, health, legal status, or irreversible systems.

## Block by default

Block actions that:

- lack a clear promise,
- exceed boundary,
- hide cost,
- create records without consent,
- treat a person as a score,
- escalate conflict without review,
- affect third parties without authorization,
- cannot be repaired if wrong.

## Local-first principle

Prefer local processing.
Prefer minimal disclosure.
Prefer temporary state.
Prefer user-controlled deletion.
Prefer summary over raw data.
Prefer human review over silent execution.

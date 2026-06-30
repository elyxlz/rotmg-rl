# Dev launchers (box-specific)

The exact scripts used to train and verify on the development GPU box — they carry hardcoded box
paths and tuned env (CPU pinning, `SIM_*` flags, redis ports). They document *how the ~99% run was
actually produced*; they are **not** portable entry points.

For a clean, portable reproduce use `scripts/reproduce.sh` (and `scripts/setup.sh`) instead.

- `run-longrun.sh` — the real long curriculum run (server-as-sim, d 0→1, the wired reward env).
- `run-learn.sh`, `sweep-async.sh` — the Protein sweep + learning launchers.
- `run-async-smoke.sh`, `run-async-tput.sh`, `run-lockstep-tput.sh` — shm-mode smoke + throughput checks.
- `run-verify-obs.sh`, `run_combat_proof.sh` — the fidelity / combat verification harness.
- `bench_fight.py`, `bench_gate.py` — micro-benchmarks for the fight loop + the shm gate.
- `curve_summary.py` — renders the learning-curve CSV under `proof/`.

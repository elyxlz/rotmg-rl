# archive

Superseded scripts, kept for reference. The training stack consolidated onto **PufferLib 4.0**
(`train.py` at the repo root + `scripts/*4*`). These are the old **PufferLib 3.0 (PuffeRL)** training
/ eval / sweep / bench tooling and the 6-subprocess curriculum, replaced by the single-process
`train.py`.

Not imported by any active code or the deploy (the deploy uses `src/rotmg_rl/csim/` + `deploy/v3/`,
which are NOT archived). Restore with `git mv scripts/archive/<f> scripts/<f>` if needed.

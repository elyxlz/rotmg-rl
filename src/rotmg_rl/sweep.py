"""Protein sweep search space + the hparam plumbing the server-as-sim trainer shares.

`build_sweep_config` defines the Protein search space (cost-aware Bayesian over CURRICULUM DEPTH -- how
far up the difficulty ladder the policy clears); `_hparams_from_args` / `apply_hparams` move the swept
knobs between the flat hp dict and the pufferl config sections. The server-as-sim driver lives in
`rotmg_rl.trainer.sweep`; this module owns only the shared search-space definition.
"""

from __future__ import annotations

# Swept hyperparameters, by the pufferl-config section they live in. ramp_frac is schedule-only
# (PuffeRL ignores it; train_continuous reads it). Every key here is verified to be read by 4.0
# PuffeRL / load_policy / the env Config -- no dead dimensions (update_epochs/vtrace are NOT read; we
# don't sweep horizon since minibatch_size % horizon must hold, nor the clip coeffs per the guide).
SWEEP_TRAIN = ("learning_rate", "gamma", "gae_lambda", "ent_coef", "vf_coef", "max_grad_norm", "minibatch_size", "ramp_frac")
SWEEP_POLICY = ("hidden_size", "num_layers")
SWEEP_ENV = ("rew_approach", "rew_boss_dmg", "rew_clear", "rew_speed", "rew_death", "rew_step")
_INT_HP = ("hidden_size", "num_layers", "minibatch_size")


def apply_hparams(args: dict, hp: dict) -> None:
    """Write whatever swept knobs are present in hp into the right pufferl-config section in place."""
    for section, keys in (("train", SWEEP_TRAIN), ("policy", SWEEP_POLICY), ("env", SWEEP_ENV)):
        for k in keys:
            if k in hp:
                args[section][k] = int(hp[k]) if k in _INT_HP else hp[k]


def _hparams_from_args(args) -> dict:
    """Pull the swept knobs back out of a (suggest-filled) pufferl args dict into the flat hp dict that
    apply_hparams + train_continuous consume."""
    hp = {k: args["train"][k] for k in SWEEP_TRAIN}
    hp.update({k: args["policy"][k] for k in SWEEP_POLICY})
    hp.update({k: args["env"][k] for k in SWEEP_ENV})
    for k in _INT_HP:
        hp[k] = int(hp[k])
    return hp


def _space(distribution, lo, hi, mean):
    return {"distribution": distribution, "min": lo, "max": hi, "mean": mean, "scale": "auto"}


def build_sweep_config(metric: str = "curriculum_depth") -> dict:
    """Protein search space (16 knobs) -- give Protein the heavy lifting, few hand-set assumptions.
    Every knob is read by 4.0 PuffeRL / load_policy / the env Config (verified). gamma keeps the
    long-horizon attention (the lever that broke the 73% plateau); ramp_frac is the schedule's own knob.
    NOTE: 16 dims wants more than the default trial budget -- raise --sweep-trials (~40+) for a thorough
    search; 16-24 is a coarse pass. We deliberately do NOT sweep the clip coefficients (guide warning)
    nor horizon (minibatch_size % horizon must hold)."""
    return {
        "method": "Protein",
        "metric": metric,
        "metric_distribution": "linear",  # curriculum depth is a linear 0..1 objective
        "goal": "maximize",
        "downsample": 5,
        "max_suggestion_cost": 3600,
        "early_stop_quantile": 0.3,
        "train": {
            "learning_rate": _space("log_normal", 2e-4, 5e-2, 1.5e-2),
            "gamma": _space("logit_normal", 0.95, 0.999, 0.97),
            "gae_lambda": _space("logit_normal", 0.80, 0.99, 0.88),
            "ent_coef": _space("log_normal", 5e-4, 8e-2, 2e-2),
            "vf_coef": _space("log_normal", 0.3, 3.0, 1.0),
            "max_grad_norm": _space("log_normal", 0.3, 5.0, 1.0),
            "minibatch_size": _space("uniform_pow2", 512, 2048, 1024),  # stays divisible by horizon 64
            "ramp_frac": _space("uniform", 0.3, 0.85, 0.6),  # schedule-only knob (PuffeRL ignores it)
        },
        "policy": {
            "hidden_size": _space("uniform_pow2", 128, 1024, 256),  # cost-aware
            "num_layers": _space("int_uniform", 1, 3, 1),
        },
        "env": {
            "rew_approach": _space("uniform", 0.0, 0.06, 0.02),
            "rew_boss_dmg": _space("uniform", 0.5, 2.0, 1.0),
            "rew_clear": _space("uniform", 0.5, 3.0, 1.0),
            "rew_speed": _space("uniform", 0.0, 0.6, 0.2),  # small fast-clear bonus, tuned alongside the clear reward
            "rew_death": _space("uniform", 0.1, 1.0, 0.5),
            "rew_step": _space("uniform", -0.003, 0.0, -0.001),
        },
    }

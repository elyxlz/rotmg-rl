"""Recurrent PPO trainer for Snake Pit (CleanRL-style, over PufferLib vector envs).

Smoke (verify it learns the shrunk-boss case, ~1M steps):
    uv run --extra train python scripts/train.py --boss-hp 120 --total-timesteps 1000000 \
        --name m2-smoke --backend serial

Full run scales num-envs + Multiprocessing backend. Clear-rate here is a terminal-reward
proxy; the ground-truth M3 eval lives in scripts/eval_policy.py.
"""

from __future__ import annotations

import argparse
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pufferlib.emulation as emu
import pufferlib.vector as vector
import wandb
from rotmg_rl.observation import GRID_SHAPE
from rotmg_rl.policy import Agent
from rotmg_rl.sim.snakepit import SnakePitConfig, SnakePitEnv

GRID_FLAT = int(np.prod(GRID_SHAPE))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="m2")
    p.add_argument("--project", default="rotmg-rl")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--backend", choices=["serial", "multiprocessing"], default="serial")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=2)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--boss-hp", type=float, default=120.0)
    p.add_argument("--video-interval", type=int, default=10, help="record a rollout every N updates")
    return p.parse_args()


def flatten_obs(d: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([d["grid"].ravel(), d["scalars"]]).astype(np.float32)


@torch.no_grad()
def record_rollout(agent: Agent, cfg: SnakePitConfig, device, seed: int = 0) -> tuple[list, bool, float]:
    env = SnakePitEnv(cfg, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    lstm_state = agent.initial_state(1, device)
    done = torch.zeros(1, device=device)
    frames, total, cleared = [], 0.0, False
    for _ in range(cfg.max_steps):
        x = torch.tensor(flatten_obs(obs), device=device).unsqueeze(0)
        action, lstm_state = agent.act_greedy(x, lstm_state, done)
        obs, reward, term, trunc, info = env.step(action[0].cpu().numpy())
        frames.append(env.render().transpose(2, 0, 1))  # wandb wants (T,C,H,W)
        total += reward
        done = torch.tensor([float(term or trunc)], device=device)
        if term or trunc:
            cleared = info["cleared"]
            break
    return frames, cleared, total


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_updates = args.total_timesteps // batch_size

    wandb.init(project=args.project, name=args.name, config=vars(args))

    cfg = SnakePitConfig(boss_hp_max=args.boss_hp)
    backend = vector.Multiprocessing if args.backend == "multiprocessing" else vector.Serial
    vec_kwargs = {"num_workers": args.num_workers} if args.backend == "multiprocessing" else {}
    venv = vector.make(
        emu.GymnasiumPufferEnv,
        env_kwargs={"env_creator": partial(SnakePitEnv, cfg)},
        num_envs=args.num_envs,
        backend=backend,
        **vec_kwargs,
    )
    obs_dim = venv.single_observation_space.shape[0]
    act_dtype = venv.single_action_space.dtype

    agent = Agent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros(args.num_steps, args.num_envs, obs_dim, device=device)
    actions = torch.zeros(args.num_steps, args.num_envs, 2, dtype=torch.long, device=device)
    logprobs = torch.zeros(args.num_steps, args.num_envs, device=device)
    rewards = torch.zeros(args.num_steps, args.num_envs, device=device)
    dones = torch.zeros(args.num_steps, args.num_envs, device=device)
    values = torch.zeros(args.num_steps, args.num_envs, device=device)

    global_step = 0
    start = time.perf_counter()
    next_obs_np, _ = venv.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs_np, device=device)
    next_done = torch.zeros(args.num_envs, device=device)
    lstm_state = agent.initial_state(args.num_envs, device)

    ep_return = np.zeros(args.num_envs, np.float32)
    ep_len = np.zeros(args.num_envs, np.int64)
    recent_returns: list[float] = []
    recent_lengths: list[int] = []
    recent_clears: list[float] = []

    for update in range(1, num_updates + 1):
        initial_lstm_state = (lstm_state[0].clone(), lstm_state[1].clone())
        frac = 1.0 - (update - 1.0) / num_updates
        optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value, lstm_state = agent.get_action_and_value(next_obs, lstm_state, next_done)
            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            act_np = action.cpu().numpy().astype(act_dtype)
            next_obs_np, reward, term, trunc, info = venv.step(act_np)
            done = np.logical_or(term, trunc)
            rewards[step] = torch.tensor(reward, device=device).view(-1)
            next_obs = torch.tensor(next_obs_np, device=device)
            next_done = torch.tensor(done.astype(np.float32), device=device)

            ep_return += reward
            ep_len += 1
            for i in np.where(done)[0]:
                recent_returns.append(float(ep_return[i]))
                recent_lengths.append(int(ep_len[i]))
                recent_clears.append(1.0 if reward[i] > args.boss_hp * 0.0 + 25.0 else 0.0)  # clear bonus dominates terminal reward
                ep_return[i] = 0.0
                ep_len[i] = 0

        with torch.no_grad():
            next_value = agent.get_value(next_obs, lstm_state, next_done).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape(-1, obs_dim)
        b_actions = actions.reshape(-1, 2)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_dones = dones.reshape(-1)

        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []
        approx_kl = torch.tensor(0.0)
        for _ in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start_i in range(0, args.num_envs, envsperbatch):
                mbenvinds = envinds[start_i : start_i + envsperbatch]
                mb_inds = flatinds[:, mbenvinds].ravel()
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds],
                    (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    b_actions[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()
                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        sps = int(global_step / (time.perf_counter() - start))
        log = {
            "charts/SPS": sps,
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "losses/policy_loss": pg_loss.item(),
            "losses/value_loss": v_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
        }
        if recent_returns:
            log["charts/episodic_return"] = float(np.mean(recent_returns[-100:]))
            log["charts/episodic_length"] = float(np.mean(recent_lengths[-100:]))
            log["charts/clear_rate"] = float(np.mean(recent_clears[-100:]))

        if update % args.video_interval == 0:
            frames, cleared, total = record_rollout(agent, cfg, device, seed=update)
            log["rollout/video"] = wandb.Video(np.array(frames, np.uint8), fps=30, format="mp4")
            log["rollout/cleared"] = float(cleared)
            log["rollout/return"] = total

        wandb.log(log, step=global_step)
        if update % 10 == 0 or update == 1:
            cr = log.get("charts/clear_rate", float("nan"))
            print(f"update {update}/{num_updates} step {global_step} sps {sps} clear_rate {cr:.2f}")

    venv.close()
    torch.save(agent.state_dict(), f"checkpoints/{args.name}.pt")
    wandb.finish()
    print(f"saved checkpoints/{args.name}.pt")


if __name__ == "__main__":
    import pathlib

    pathlib.Path("checkpoints").mkdir(exist_ok=True)
    main()

"""Build a LIBRARY of SUCCESSFUL skill rollouts and fit the action-space envelope.

Unlike collect_scripted_skill_data.py (which fits a spread Sigma around the *live*
controller, over ALL actions), this keeps ONLY the actions from rollouts that
actually SUCCEEDED (cube cleared / target reached), up to the success moment, and
fits the absolute Gaussian (mu, Sigma) over those actions — literally "the action
space of successful pushes." The composition trainer's `envelope` provider then
explores within  a = clip(mu + eps * Sigma * u).

  uv run collect_skill_library.py --skill push --out skill_balls/push_env.pkl
  uv run collect_skill_library.py --skill base --out skill_balls/base_env.pkl
"""
import argparse
import os
from pathlib import Path

from brax import envs
import jax
import jax.numpy as jnp
import numpy as np

from train import save_params
from train_residual_mab import make_env
from envs.scripted_controllers import CONTROLLERS

NATIVE = {"base": "tidybot_navigate", "push": "tidybot_push_aside"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skill", choices=list(NATIVE), required=True)
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--noise", type=float, default=0.05,
                   help="Small action noise for action diversity within successes.")
    p.add_argument("--sigma_floor", type=float, default=0.03)
    p.add_argument("--state_dim", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    args = p.parse_args()

    env_id = NATIVE[args.skill]
    ctrl = CONTROLLERS[env_id]
    sd = args.state_dim

    env = make_env(env_id)
    env = envs.training.wrap(env, episode_length=args.steps)
    step = jax.jit(env.step)
    reset = jax.jit(env.reset)

    key = jax.random.PRNGKey(args.seed)
    key, rk = jax.random.split(key)
    s = reset(jax.random.split(rk, args.num_envs))

    @jax.jit
    def rollout(s, key):
        def stp(carry, _):
            s, key = carry
            key, nk = jax.random.split(key)
            a = ctrl(s.obs[:, :sd], s.obs[:, sd:sd + 2])
            a = jnp.clip(a + args.noise * jax.random.normal(nk, a.shape), -1.0, 1.0)
            s = step(s, a)
            return (s, key), (a, s.metrics["success"])
        (s, _), (acts, succ) = jax.lax.scan(stp, (s, key), None, length=args.steps)
        return acts, succ

    key, ck = jax.random.split(key)
    acts, succ = rollout(s, ck)
    acts = np.asarray(acts)   # (T, N, 11)
    succ = np.asarray(succ)   # (T, N)

    # Keep actions up to (and including) each env's FIRST success — i.e. the actions
    # that led to a successful push — for every env that ever succeeded.
    lib, n_succ = [], 0
    for n in range(acts.shape[1]):
        ts = np.where(succ[:, n] > 0.5)[0]
        if len(ts) == 0:
            continue
        n_succ += 1
        lib.append(acts[:ts[0] + 1, n, :])
    if not lib:
        raise SystemExit(
            f"[{args.skill}] NO successful episodes out of {args.num_envs} — the "
            "controller can't succeed here, so there's no success library to fit. "
            "Improve the controller or change the collection conditions.")

    L = np.concatenate(lib, axis=0)         # (M, 11)
    mu = L.mean(axis=0)
    sigma = np.maximum(L.std(axis=0), args.sigma_floor)

    print(f"[{args.skill}] success {n_succ}/{acts.shape[1]} envs  "
          f"library_actions={L.shape[0]}", flush=True)
    print(f"  mu   ={[round(float(x), 3) for x in mu]}", flush=True)
    print(f"  sigma={[round(float(x), 3) for x in sigma]}", flush=True)

    out = args.out or f"skill_balls/{args.skill}_env.pkl"
    Path(os.path.dirname(out) or ".").mkdir(parents=True, exist_ok=True)
    save_params(out, {
        "mu": jnp.asarray(mu),
        "sigma": jnp.asarray(sigma),
        "skill": args.skill,
        "env_id": env_id,
        "n_success_envs": int(n_succ),
        "n_total_envs": int(acts.shape[1]),
        "n_action_samples": int(L.shape[0]),
    })
    print(f"[save] {out}", flush=True)


if __name__ == "__main__":
    main()

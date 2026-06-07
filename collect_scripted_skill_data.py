"""Collect scripted-skill rollouts and fit each skill's diagonal Mahalanobis ball.

Runs a skill's hand-coded controller in its native env over randomized conditions
(with a small action noise), then fits the ball metric
    Sigma = max(per-dim std of the actions, sigma_floor)   (diagonal).
The ball CENTER is the controller (code, used live at composition time); only Sigma
("the constrained space") is saved. A diagonal 11-dim Sigma is statistically cheap,
so what matters is *diversity of conditions* (each env reset = a distinct config),
not raw sample count.

  uv run collect_scripted_skill_data.py --skill base --out skill_balls/base.pkl
  uv run collect_scripted_skill_data.py --skill push --out skill_balls/push.pkl
"""
import argparse
import os
from pathlib import Path

from brax import envs
import jax
import jax.numpy as jnp

from train import save_params
from train_residual_mab import make_env
from envs.scripted_controllers import CONTROLLERS

# skill -> (native env, arm-start noise). Base: navigate env (diverse targets ->
# isotropic base Sigma). Push: push_aside env with arm-start noise so the
# controller's drive-to-contact exercises ALL arm joints (else only j1 varies and
# the push ball would be too narrow to lower the arm from tucked in the hallway).
SKILL = {"base": ("tidybot_navigate", 0.0), "push": ("tidybot_push_aside", 0.8)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skill", choices=list(SKILL), required=True)
    p.add_argument("--num_collect_envs", type=int, default=256)
    p.add_argument("--collect_steps", type=int, default=200)
    p.add_argument("--noise", type=float, default=0.1,
                   help="Action-space exploration noise added to the controller.")
    p.add_argument("--sigma_floor", type=float, default=0.05)
    p.add_argument("--state_dim", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    args = p.parse_args()

    env_id, arm_noise = SKILL[args.skill]
    ctrl = CONTROLLERS[env_id]
    sd = args.state_dim

    env = make_env(env_id)
    if arm_noise > 0:
        env.arm_noise_scale = arm_noise  # randomize arm start -> exercise all arm dims
    env = envs.training.wrap(env, episode_length=args.collect_steps)
    step = jax.jit(env.step)
    reset = jax.jit(env.reset)

    key = jax.random.PRNGKey(args.seed)
    key, rk = jax.random.split(key)
    state = reset(jax.random.split(rk, args.num_collect_envs))

    @jax.jit
    def rollout(state, key):
        def stp(carry, _):
            state, key = carry
            key, nk = jax.random.split(key)
            goal = state.obs[:, sd:sd + 2]            # native env goal lives in the tail
            a = ctrl(state.obs[:, :sd], goal)
            a = jnp.clip(a + args.noise * jax.random.normal(nk, a.shape), -1.0, 1.0)
            state = step(state, a)
            return (state, key), a
        (state, _), acts = jax.lax.scan(
            stp, (state, key), None, length=args.collect_steps)
        return acts  # (T, N, 11)

    key, ck = jax.random.split(key)
    acts = rollout(state, ck)
    A = acts.reshape(-1, acts.shape[-1])  # (T*N, 11)
    sigma = jnp.maximum(jnp.std(A, axis=0), args.sigma_floor)

    # Split-half stability: if Sigma from the two halves matches, there's enough data.
    half = A.shape[0] // 2
    drift = float(jnp.max(jnp.abs(jnp.std(A[:half], axis=0) - jnp.std(A[half:], axis=0))))

    print(f"[{args.skill}] env={env_id} samples={A.shape[0]}", flush=True)
    print(f"  sigma={[round(float(x), 3) for x in sigma]}", flush=True)
    print(f"  base-dim(0:3) mean sigma={float(jnp.mean(sigma[:3])):.3f}  "
          f"arm-dim(3:10) mean sigma={float(jnp.mean(sigma[3:10])):.3f}  "
          f"gripper(10) sigma={float(sigma[10]):.3f}", flush=True)
    print(f"  split-half max sigma drift={drift:.4f} (small => enough data)", flush=True)

    out = args.out or f"skill_balls/{args.skill}.pkl"
    Path(os.path.dirname(out) or ".").mkdir(parents=True, exist_ok=True)
    save_params(out, {
        "sigma": sigma,
        "env_id": env_id,
        "skill": args.skill,
        "num_samples": int(A.shape[0]),
        "noise": args.noise,
        "sigma_floor": args.sigma_floor,
    })
    print(f"[save] {out}", flush=True)


if __name__ == "__main__":
    main()

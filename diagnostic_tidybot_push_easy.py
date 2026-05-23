"""Phase 1 diagnostic for tidybot_push_easy.

Three checks designed to disambiguate audit hypotheses (sparse-reward vs
NaN vs action-conversion bug):

  1. Initial state inspection (cube / goal / EEF positions and dist).
  2. Per-dim action sweep (does each action dim affect cube / base?).
  3. Random-policy rollout (NaN check + final cube-goal distance distribution).

Run with `uv run diagnostic_tidybot_push_easy.py`. Headless; no GPU required.
"""

import time

import jax
import jax.numpy as jnp

from envs.mobile_manipulation.tidybot_push_easy import TidyBotPushEasy


DIM_NAMES = [
    "base_x",  "base_y",  "base_th",
    "arm_1",   "arm_2",   "arm_3",   "arm_4",   "arm_5",   "arm_6",   "arm_7",
    "gripper",
]


def main() -> None:
    env = TidyBotPushEasy(backend="mjx")
    print(
        f"\nEnv loaded. action_size={env.action_size}, "
        f"episode_length={env.episode_length}, n_frames={env.n_frames}"
    )

    step_jit = jax.jit(env.step)
    reset_jit = jax.jit(env.reset)

    # --- 1. Initial state ---
    rng = jax.random.PRNGKey(0)
    state = reset_jit(rng)
    cube_start = state.obs[10:13]
    goal = state.info["goal"][:3]
    eef_start = state.obs[13:16]
    init_dist = float(jnp.linalg.norm(cube_start - goal))
    print("\n=== initial state (seed=0) ===")
    print(f"  cube start   (obs[10:13]): {cube_start}")
    print(f"  goal         (info[goal]):  {goal}")
    print(f"  eef start    (obs[13:16]): {eef_start}")
    print(f"  cube-goal dist (initial):   {init_dist:.4f} m")

    # --- 2. Per-dim action sweep ---
    print("\n=== per-dim action sweep (50 steps, hold one dim at ±1.0) ===")
    print(
        f"  {'dim':>3s} {'name':>9s} "
        f"{'cube Δ +1.0':>13s} {'cube Δ -1.0':>13s} "
        f"{'base Δ +1.0':>13s} {'base Δ -1.0':>13s}"
    )
    for dim in range(11):
        deltas = []
        for sign in (1.0, -1.0):
            s = reset_jit(rng)
            cube0 = s.obs[10:13]
            base0 = s.obs[0:3]
            action = jnp.zeros(11).at[dim].set(sign)
            for _ in range(50):
                s = step_jit(s, action)
            cube_d = float(jnp.linalg.norm(s.obs[10:13] - cube0))
            base_d = float(jnp.linalg.norm(s.obs[0:3] - base0))
            deltas.append((cube_d, base_d))
        print(
            f"  {dim:>3d} {DIM_NAMES[dim]:>9s} "
            f"{deltas[0][0]:>11.4f} m {deltas[1][0]:>11.4f} m "
            f"{deltas[0][1]:>11.4f} m {deltas[1][1]:>11.4f} m"
        )

    # --- 3. Random rollouts ---
    print("\n=== random-policy rollouts (100 episodes × 50 steps) ===")
    nan_episodes = 0
    nan_first_steps = []
    any_reward_episodes = 0
    final_dists = []

    t0 = time.time()
    for seed in range(100):
        rng_ep = jax.random.PRNGKey(seed)
        s = reset_jit(rng_ep)
        ep_nan_at = -1
        ep_has_reward = False
        for t in range(50):
            rng_ep, key = jax.random.split(rng_ep)
            action = jax.random.uniform(key, (11,), minval=-1.0, maxval=1.0)
            s = step_jit(s, action)
            if ep_nan_at < 0:
                obs_nan = bool(jnp.any(jnp.isnan(s.obs)))
                rew_nan = bool(jnp.isnan(s.reward))
                if obs_nan or rew_nan:
                    ep_nan_at = t
            if float(s.reward) > 0:
                ep_has_reward = True
        if ep_nan_at >= 0:
            nan_episodes += 1
            nan_first_steps.append(ep_nan_at)
        if ep_has_reward:
            any_reward_episodes += 1
        final_dists.append(float(jnp.linalg.norm(s.obs[10:13] - s.info["goal"][:3])))
    elapsed = time.time() - t0

    final_dists_arr = jnp.array(final_dists)
    print(f"\nResults (in {elapsed:.1f}s):")
    print(f"  NaN episodes:                   {nan_episodes}/100")
    if nan_first_steps:
        print(f"  first NaN step (mean):          {sum(nan_first_steps)/len(nan_first_steps):.1f}")
    print(f"  episodes with any reward>0:     {any_reward_episodes}/100")
    print(f"  cube-goal dist (final, mean):   {float(final_dists_arr.mean()):.3f} m")
    print(f"  cube-goal dist (final, min):    {float(final_dists_arr.min()):.3f} m")
    print(f"  cube-goal dist (final, max):    {float(final_dists_arr.max()):.3f} m")
    print(f"  initial cube-goal dist (ref):   {init_dist:.4f} m")

    # Reward threshold for success is dist < 0.1 in the env code.
    fraction_within_threshold = float(jnp.mean(final_dists_arr < 0.1))
    print(f"  fraction with final dist <0.1m: {fraction_within_threshold:.3f}")


if __name__ == "__main__":
    main()

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
import numpy as np

from envs.mobile_manipulation.tidybot_push_easy import TidyBotPushEasy


DIM_NAMES = [
    "base_x",  "base_y",  "base_th",
    "arm_1",   "arm_2",   "arm_3",   "arm_4",   "arm_5",   "arm_6",   "arm_7",
    "gripper",
]

# Reset arm pose (arm_q_default in TidyBotPushEasy._get_initial_state, arm_noise=0).
ARM_DEFAULT = np.array([-np.pi / 2, 1.4, np.pi, -1.2, 0.0, 0.5, np.pi / 2])


def _arm_q(state):
    return np.array(state.obs[3:10])


def arm_tracking_test(env, reset_jit, step_jit) -> None:
    """Verify the (delta-control) arm goes where commanded.

    The env's delta control is measurement-relative: target = current + action*0.25*mult.
    So a *zero* action does NOT firmly hold against gravity (it re-targets the drooping
    current pose, so the arm sags). The correct way to reach/hold a pose is to actively
    command it: action = (q_tgt - current) / (0.25 * multiplier).
    Checks: reset pose, zero-action drift (informational), active hold, reach, direction.
    """
    print("\n=== arm-tracking test (delta control: action 0 = hold) ===")
    arm_min = np.array(env.arm_joint_range[:, 0])
    arm_max = np.array(env.arm_joint_range[:, 1])
    arm_mult = (arm_max - arm_min) / 2.0

    def delta_action(q_tgt, arm_q):
        a = np.zeros(11)
        a[3:10] = np.clip((q_tgt - arm_q) / (0.25 * arm_mult), -1.0, 1.0)
        return jnp.array(a)

    rng = jax.random.PRNGKey(0)

    # Reset pose should match ARM_DEFAULT.
    state = reset_jit(rng)
    q0 = _arm_q(state)
    reset_err = float(np.max(np.abs(q0 - ARM_DEFAULT)))
    print(f"  reset pose  max|q-ARM_DEFAULT| = {reset_err:.4f} rad  "
          f"[{'OK' if reset_err < 0.1 else 'FAIL'}]")
    assert reset_err < 0.1, f"reset arm pose off by {reset_err:.3f} rad"

    # Zero-action drift (informational): measurement-relative delta control does not
    # reject gravity, so the arm sags. Sanity-check it stays finite and bounded.
    s = reset_jit(rng)
    for _ in range(50):
        s = step_jit(s, jnp.zeros(11))
    zero_drift = float(np.max(np.abs(_arm_q(s) - q0)))
    zero_nan = bool(jnp.any(jnp.isnan(s.obs)))
    print(f"  zero-action drift over 50 steps  = {zero_drift:.4f} rad  "
          f"(informational; sags under gravity)")
    assert not zero_nan and zero_drift < 2.0, "zero-action arm went NaN or unbounded"

    # Active hold: continuously commanding the start pose should hold it (rejects gravity).
    s = reset_jit(rng)
    for _ in range(50):
        s = step_jit(s, delta_action(q0, _arm_q(s)))
    hold_err = float(np.max(np.abs(_arm_q(s) - q0)))
    print(f"  active hold max|q-q0| (50 steps) = {hold_err:.4f} rad  "
          f"[{'OK' if hold_err < 0.15 else 'FAIL'}]")
    assert hold_err < 0.15, f"active hold failed: {hold_err:.3f} rad"

    # Reach: drive to several absolute target poses; check convergence. A position
    # servo without gravity compensation has bounded steady-state error under load
    # (joints near target get ~0 action, which sags until the error self-corrects),
    # so "approximately reached" uses a ~0.2 rad (~11 deg) tolerance.
    REACH_TOL = 0.2
    targets = {
        "ARM_LOWER": np.array([-np.pi / 2, 1.9, np.pi, -1.0, 0.0, 0.5, np.pi / 2]),
        "interior":  np.array([0.3, 1.0, 0.5, -1.5, -0.5, 0.8, 0.6]),
    }
    worst = 0.0
    for name, q_tgt in targets.items():
        s = reset_jit(rng)
        for _ in range(150):
            s = step_jit(s, delta_action(q_tgt, _arm_q(s)))
        err = np.abs(_arm_q(s) - q_tgt)
        max_err = float(np.max(err))
        worst = max(worst, max_err)
        print(f"  reach {name:>9s}  max|q-q_tgt| = {max_err:.4f} rad  "
              f"(worst joint arm_{int(np.argmax(err)) + 1})  [{'OK' if max_err < REACH_TOL else 'FAIL'}]")
    assert worst < REACH_TOL, f"arm failed to reach a target: worst err {worst:.3f} rad"

    # Direction: +action on arm_2 (headroom above ARM_DEFAULT) should raise the angle.
    s = reset_jit(rng)
    j = 1  # arm_2
    a = jnp.zeros(11).at[3 + j].set(0.5)
    q_before = float(_arm_q(s)[j])
    for _ in range(30):
        s = step_jit(s, a)
    q_after = float(_arm_q(s)[j])
    print(f"  direction arm_2 (+action) = {q_before:.3f} -> {q_after:.3f} rad  "
          f"[{'OK' if q_after > q_before + 0.05 else 'FAIL'}]")
    assert q_after > q_before + 0.05, "arm_2 did not increase with +action"

    print("  arm-tracking test: PASS")


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

    # --- 1b. Arm tracking (delta control) ---
    arm_tracking_test(env, reset_jit, step_jit)

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

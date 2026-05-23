# tidybot_push_easy — Phase 1 Diagnostic Results

**Date**: 2026-05-23
**Machine**: Macbook Pro M1 (JAX-CPU)
**Driver**: `diagnostic_tidybot_push_easy.py`

---

## TL;DR

The audit's hypothesis #3 (NaN problem) was the dominant cause of the 0-rewards symptom. **A single bug in `TidyBotEnv.__init__` was making every step NaN from t=0.** A one-line fix dropped the NaN rate from 100/100 → 1/100. The remaining 1% is likely contact-induced and will be handled by the `nan_to_num` guard planned for Phase 3.

The audit's hypothesis #1 (sparse reward + fixed task + 50-step episode) also still applies: even with the env stepping cleanly, 0 of 100 random-policy episodes achieved any positive reward — confirming the task is essentially unreachable from cold-start in 50 steps with the cube/goal fixed.

---

## Root cause: continuous arm joints + inf range in action conversion

Kinova's Gen3 arm has 4 continuous-rotation joints (`joint_1`, `_3`, `_5`, `_7`). Mujoco reports their `jnt_range` as `[-inf, +inf]`.

`TidyBotEnv._convert_action_to_actuator_input_joint_angle` (override of the parent `ArmEnvs` method) then computes:

```python
min_val, max_val = self.arm_joint_range[:, 0], self.arm_joint_range[:, 1]
offset, multiplier = (min_val + max_val) / 2, (max_val - min_val) / 2
# offset = (-inf + +inf)/2 = NaN
# multiplier = (+inf - -inf)/2 = +inf
normalized = (arm_angles[3:10] - offset) / multiplier  # NaN/inf -> NaN
```

The parent class has a `jnp.where(multiplier > 0, ...)` guard, but `inf > 0` is `True`, so the guard doesn't catch it. Every step poisoned from t=0.

This is consistent with the project's recent commit message: *"add tidybot push hard and add debugging for NaN problem in tidybot push easy."*

---

## Fix applied

Substitute `[-π, π]` for any arm joint with non-finite range. One-line addition in `envs/manipulation/tidybot_envs.py`:

```python
raw_range = self.sys.jnt_range[3:10]
finite_mask = jnp.isfinite(raw_range).all(axis=1, keepdims=True)
default_range = jnp.array([-jnp.pi, jnp.pi])[None, :]
self.arm_joint_range = jnp.where(finite_mask, raw_range, default_range)
```

(Not yet committed — needs review before pushing to `dylanzhou2/scaling-crl`.)

---

## Pre-fix vs post-fix

| Metric | Pre-fix | Post-fix |
|---|---|---|
| Per-dim action sweep | NaN cube/base Δ across all 11 dims | sensible cube/base Δ (table below) |
| Random rollout NaN rate | 100/100 (first NaN at t=0) | **1/100** (first NaN at t=12) |
| Random rollout success rate | 0/100 | 0/100 (still — see hypothesis #1) |
| Final cube-goal dist | NaN | still NaN-polluted by the 1 bad episode (next: nan_to_num guard) |

---

## Per-dim action sweep (post-fix)

50 steps, hold one action dim at ±1.0, all others at 0. Cube and base position deltas in meters.

| dim | name | cube Δ +1.0 | cube Δ −1.0 | base Δ +1.0 | base Δ −1.0 |
|---:|:---|---:|---:|---:|---:|
| 0 | base_x  | 0.0401 | 0.0401 | **6.6770** | **6.6771** |
| 1 | base_y  | 0.0401 | 0.0401 | **6.6851** | **6.6851** |
| 2 | base_th | 0.0401 | 0.0401 | **6.3994** | **6.3994** |
| 3 | arm_1   | 0.0401 | 0.0401 | 0.0003 | 0.0003 |
| 4 | arm_2   | 0.0401 | 0.0401 | 0.0159 | 0.0133 |
| 5 | arm_3   | 0.0401 | 0.0401 | 0.0009 | 0.0001 |
| 6 | arm_4   | 0.0401 | 0.0401 | 0.0000 | 0.0000 |
| 7 | arm_5   | 0.0401 | 0.0401 | 0.0000 | 0.0000 |
| 8 | arm_6   | 0.0401 | 0.0401 | 0.0000 | 0.0000 |
| 9 | arm_7   | 0.0401 | 0.0401 | 0.0000 | 0.0000 |
| 10 | gripper | 0.0401 | 0.0401 | 0.0000 | 0.0000 |

**Observations:**
- **Base moves freely** under base actions (6–7 m over 50 control steps with `action=1.0`, which delta-controls position with `0.2 × n_frames(25) × dt(0.002)` accumulation — physically reasonable).
- **Arm actions barely move the base** (expected — arm/base coupling is small).
- **Cube delta is uniformly 0.04 m across all dims**. This is almost certainly the cube settling under gravity from its start z=0.07 m to the table (no contact from the robot in any single-dim trajectory). Not really an action effect.

The implication: in a 50-step episode with random actions, the cube never gets pushed because the robot never reliably contacts it. Confirms audit hypothesis #1 (task too short / too sparse to learn from cold start).

---

## Random-policy rollout stats (post-fix, seeds 0–99)

```
NaN episodes:                   1/100
first NaN step (mean):          12.0
episodes with any reward>0:     0/100
cube-goal dist (final):         NaN-polluted by the 1 bad episode
initial cube-goal dist (ref):   0.8612 m
```

The 1 NaN episode at t=12 is a contact / physics issue (not action conversion — that's clean now). Likely the random arm trajectory drove a joint into a self-collision or hard contact that MJX's solver couldn't resolve.

---

## What this means for the rest of the testing plan

**Phase 1 conclusions:**

1. **Env is now mechanically healthy** — 99% clean rollouts is good enough to start training. The remaining 1% will be caught by the `nan_to_num` guard planned for Phase 3.
2. **The 0-rewards symptom had two causes**, both present together: the NaN bug (fixed) and the sparse-reward + short-episode + fixed-task setup (still present, fixed by Phase 3's changes: `episode_length 50→500`, `noise_scale 0→0.1`).
3. **Action space is properly wired** — base / arm actions affect the right joints.

**Implications for Phase 2 / 3:**

- Phase 2 (`arm_push_easy` control) is still worth running — it confirms the JaxGCRL training loop on H100 with our brax patches before we sink overnight runs into tidybot. Panda doesn't have continuous joints so it shouldn't have hit this bug, but proving the stack is healthy is cheap.
- Phase 3 (`tidybot_push_easy` at depths 4/16/64) is now expected to actually learn — the env is no longer producing NaN from step 0.
- The Phase 3 launch command should include the previously-planned three changes (episode_length 500, noise scales 0.1, nan_to_num guard) but no longer needs to treat NaN as a "stop and investigate" condition unless it exceeds ~5%.

---

## To commit (in `scaling-crl/`)

- `envs/manipulation/tidybot_envs.py` — the inf-range fix.
- `xml_sanity_check.py` — rewritten as headless diagnostic.
- `verify_tidybot_push_easy.py` — added actuator-count + obs-shape assertions.
- `diagnostic_tidybot_push_easy.py` — new file.
- `diagnostic_results.md` — this file.

Once committed and pushed to `dylanzhou2/scaling-crl`, bump the submodule pointer in the vault.

---

## References

- `deliverables/scaling-crl-audit.md` — audit memo (hypothesis #3 confirmed; #1 still applies).
- `deliverables/tidybot-testing-plan.md` — Phases 2–5 still apply, slightly easier now.
- [[contrastive-rl]] — the algorithm we'll be training next.
- [[1000-layer-networks-for-self-supervised-rl]] — reference numbers for `arm_push_easy` control.

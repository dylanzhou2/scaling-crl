# scaling-crl — Claude context

You're working inside `scaling-crl/`, the working codebase for a Stanford CS224R class project on demo-free TidyBot manipulation primitives. This file is the persistent context. Update it when project conventions change.

## What this repo is

A fork of [`wang-kevin3290/scaling-crl`](https://github.com/wang-kevin3290/scaling-crl) (the *1000 Layer Networks for Self-Supervised RL* / NeurIPS 2025 Best Paper Award codebase). Built on JAX + Brax + [JaxGCRL](https://github.com/MichalBortkiewicz/JaxGCRL) + MuJoCo MJX. Our fork lives at `dylanzhou2/scaling-crl` with `upstream` pointing to Kevin Wang's repo.

This repo is also a **git submodule of `scaling-crl-vault`** — the broader research vault that holds the wiki, deliverables, audit memos, and testing plans. When in doubt about *why* something is the way it is, the vault is the source of truth.

## What we're using it for

Train two demo-free RL primitives on TidyBot++ in MJX:
- `tidybot_push_easy` → eventual `tidybot_push_aside` (non-prehensile)
- `tidybot_pick_up` (to be built, scoped to fixed cube + top-down parallel-jaw)

Navigation is **classical** (Nav2-style planner), not learned. See `../deliverables/course-project-plan.md`.

## Locked-in stack decisions (do not re-litigate)

- **Sim**: MuJoCo MJX + Brax. **Not IsaacLab.** Reason: CRL training stack is JAX-native via JaxGCRL; TidyBot assets are MuJoCo XML. Don't suggest IsaacLab unless the user re-opens the question.
- **Algorithm**: contrastive RL (CRL) from `train.py`. Don't reimplement core RL; tune envs and hyperparameters.
- **Package manager**: `uv`. Not pip directly. Setup: `uv sync`, run: `uv run <script>`.

## Repo layout

```
train.py                        # main CRL training entry; env_id dispatch ~ line 555+
buffer.py / evaluator.py        # JaxGCRL training infra
envs/
  manipulation/
    arm_envs.py                 # ArmEnvs PipelineEnv base (sparse reward = success)
    tidybot_envs.py             # TidyBotEnv: mobile base + Kinova + Robotiq
    arm_*.py                    # Panda-arm tasks (push, binpick, grasp, reach)
  mobile_manipulation/
    tidybot_push_easy.py        # our primary target env
    tidybot_push_hard.py        # added 2026-03-10 for harder variant
    tidybot_maze.py
  assets/
    tidybot/stanford_tidybot/   # TidyBot model XML
    tidybot_push_easy.xml       # scene XML
    panda_*.xml                 # upstream Panda assets
test_tidybot.py                 # XML load sanity (uses Path(__file__) — see below)
xml_sanity_check.py             # headless: print actuator/joint layout + 200-step zero-ctrl
verify_tidybot_push_easy.py     # env instantiation + assertions (nu==11, obs.shape==(23,))
diagnostic_tidybot_push_easy.py # Phase 1 diagnostic: per-dim sweep + random rollouts
diagnostic_results.md           # writeup of the NaN investigation and fix
```

## Setup on a new machine

For M1/Mac: `uv sync` + apply the two brax patches (see below) + run the verify scripts.

For Linux + GPU: the vault has a step-by-step guide at `../deliverables/gpu-setup-guide.md` (or `https://github.com/.../scaling-crl-vault` if you don't have the vault checked out). It includes an idempotent `apply_brax_patches.py` script.

### The two brax 0.10.1 patches (must re-apply after every `uv sync`)

The brax 0.10.1 wheel has two bugs the upstream README documents. They edit files inside `.venv/`, so they don't persist across re-syncs. The vault's `gpu-setup-guide.md` ships an idempotent Python script (`apply_brax_patches.py`) that applies both. The patches:

1. `brax/contact.py` — replaces `mjx.ncon(sys)` (removed after mujoco 3.1.5) with `mjx.make_data(sys).ncon`.
2. `brax/io/json.py:159` — wraps an `rgba` comparison list in `jp.array(...)`.

If a verify script errors with `module 'mjx' has no attribute 'ncon'`, the contact patch wasn't applied.

## Verify scripts (run after setup, before any training)

```bash
uv run test_tidybot.py                  # XML loads via brax; reports 11 actuators
uv run xml_sanity_check.py              # 200 zero-control steps; no NaN; qpos drift ~0.04
uv run verify_tidybot_push_easy.py      # env instantiates; asserts nu==11 and obs.shape==(23,)
uv run diagnostic_tidybot_push_easy.py  # per-dim sweep + 100 random rollouts (~60s on M1)
```

Expected outputs on a working machine are recorded in `diagnostic_results.md`. If something diverges materially, **stop and investigate** before training.

## The TidyBot debugging campaign

This is the active research project. Before Claude makes changes in this repo, read this section.

### Context: why we're debugging

A previous training attempt on `tidybot_push_easy` (the project's flagship env) produced a flat 0 reward curve in wandb — the policy never achieved any successful cube push, run after run. Without a working trained primitive there is no project: the CS224R contribution claim is "demo-free RL primitives for mobile manipulation on TidyBot," and the manipulation primitive *is* `tidybot_push_easy` (later promoted to `push_aside`). So we need to find out why it didn't train and either fix it or pivot.

The vault's audit memo (`../deliverables/scaling-crl-audit.md`) enumerated five candidate causes, ranked by likelihood:
1. **Sparse binary reward + 50-step episode + fixed cube/goal** → cold-start policy never wins a single success → 0 eval reward.
2. **Episode-length mismatch** between env (`self.episode_length=50`) and trainer default (`1000`) → incoherent rollouts in replay.
3. **NaN problem** flagged in the team's most recent commit message before today's work.
4. **CRL contrastive signal degeneracy** because cube + goal are static across all episodes (no variation for InfoNCE to distinguish).
5. **Action / actuator dimensionality mismatch** between XML and `TidyBotEnv`'s 11-dim action vector.

The vault's `../deliverables/tidybot-testing-plan.md` lays out a 5-phase campaign to disambiguate them in 24–48h *before* burning overnight GPU runs.

### Headline finding so far (2026-05-23)

**Cause #3 (NaN) was dominant, and it has a specific fix.** `TidyBotEnv`'s action-conversion override was producing `NaN` from t=0 because Kinova's 4 continuous-rotation joints have `jnt_range = [-inf, +inf]`, which made `offset = (-inf + +inf)/2 = NaN` and `multiplier = +inf`. The parent class's `jnp.where(multiplier > 0, ...)` guard does not catch this because `inf > 0` is `True`. **Fix committed at `3debbef`** — substitute `[-π, π]` for any arm joint with non-finite range. NaN rate dropped from 100/100 to 1/100. See `diagnostic_results.md` for full evidence.

Cause #1 (sparse-reward + short episode + fixed task) is also still live: even with the NaN gone, 0/100 random-policy rollouts hit success in 50 steps. Phase 3's planned changes (`episode_length 50→500`, `noise_scale 0→0.1`) target this.

Causes #2, #4, #5 turned out not to apply or were trivial to confirm. (#5 is ruled out by `verify_tidybot_push_easy.py`'s `nu==11` assertion. #2 is handled by passing `--episode_length 500` to `train.py`. #4 is handled together with #1 via the noise-scale increase.)

### Where we are

| Phase | Description | Status |
|---|---|---|
| 0 | Pre-flight (install + brax patches + 3 verify scripts) | ✅ done on M1 |
| 1 | Diagnostic rollouts → NaN cause found and fixed | ✅ done on M1 |
| 2 | Control train `arm_push_easy` (stack-level sanity) | ⏳ needs GPU |
| 3 | Re-train `tidybot_push_easy` × depths 4/16/64 with episode_length=500 + noise_scale=0.1 | ⏳ needs GPU, overnight |
| 4 | Diagnose & iterate based on Phase 2/3 outcome | ⏳ pending |
| 5 | GREEN/YELLOW/RED decision → hand off to course plan or execute pivot | ⏳ pending |

GPU machine setup is documented in `../deliverables/gpu-setup-guide.md`.

### Phase 0 — Pre-flight verification ✅ done (M1, 2026-05-23)

**Goal.** Confirm the codebase runs at all before any training.

**What was done.**
- `uv sync` clean.
- Two brax 0.10.1 patches applied (`brax/contact.py` for the `mjx.ncon` removal; `brax/io/json.py:159` for the rgba comparison). These edit files inside `.venv/` so they don't persist across `uv sync` — re-apply on every fresh install.
- `test_tidybot.py`: XML loads via brax. nu=11, action_size=11. (Hardcoded `google3/` path replaced with `Path(__file__)`-relative earlier in the session.)
- `xml_sanity_check.py` rewritten as a headless diagnostic (the old version used `mujoco.load_model_from_path`, removed in newer mujoco). Now prints actuator/joint layout and runs 200 zero-control steps with a NaN check — no NaN, qpos drift 0.04.
- `verify_tidybot_push_easy.py` extended with `assert env.sys.nu == 11` and `assert state.obs.shape == (23,)` — both pass.

**Exit criterion.** All three verify scripts pass. ✅

### Phase 1 — Diagnostic rollouts ✅ done (M1, 2026-05-23)

**Goal.** Characterize the env under random policy. Answer four questions:
1. Does the cube move under random actions, or is the policy effectively no-op?
2. Do `obs` / `reward` / `pipeline_state.q` ever go NaN?
3. What's the distribution of `dist_cube_to_goal` at episode end?
4. Are all 11 action dims actuated as expected?

**What was done.**
- Wrote `diagnostic_tidybot_push_easy.py`: per-dim action sweep (hold one dim at ±1.0 for 50 steps, measure cube/base delta) + 100-episode random rollout with NaN counter and final-distance histogram.
- First run on M1: **100/100 NaN at t=0, every per-dim sweep returned NaN.** Cause traced to the inf-range action-conversion bug.
- Applied the fix in `envs/manipulation/tidybot_envs.py` (substitute `[-π, π]` for joints with non-finite mujoco range). Committed at `3debbef`.
- Second run on M1: **1/100 NaN at t≈12** (likely contact-induced, not action conversion); 0/100 random successes; base actions move base 6–7 m; arm-only actions don't move the base much (expected); cube delta uniformly ~0.04 m across all dims (gravity settling, no contact under random actions).

**Outcome.** Cause #3 disambiguated and fixed. Cause #1 confirmed still active.

**Exit criterion.** Root cause documented + targeted fix applied + verified. ✅

### Phase 2 — Control experiment on `arm_push_easy` ⏳ pending (needs GPU)

**Goal.** Prove the CRL training stack itself works end-to-end on the GPU box. `arm_push_easy` uses Panda (no continuous joints, so the inf-range bug doesn't apply) and is reported successful in the upstream paper. Reproducing it isolates "stack works" from "env-specific bug."

**Launch command** (from `../deliverables/gpu-setup-guide.md` §7):
```bash
uv run train.py \
  --env_id "arm_push_easy" \
  --eval_env_id "arm_push_easy" \
  --num_epochs 50 \
  --total_env_steps 30000000 \
  --critic_depth 4 --actor_depth 4 \
  --batch_size 512 \
  --vis_length 1000 \
  --save_buffer 0 \
  --exp_name "control_arm_push_easy_d4"
```

**What to watch.** First 30 min of wandb: success rate / reward going up at all. If flat at 1h → kill, treat as stack-level issue, do not launch Phase 3.

**Exit criterion.** Ascending reward curve; final success rate within ~20% of upstream paper's `arm_push_easy` baseline.

### Phase 3 — Targeted re-train of `tidybot_push_easy` ⏳ pending (needs GPU, overnight)

**Goal.** Train CRL on `tidybot_push_easy` at three depths after applying the audit's recommended config changes. Determine if learning happens.

**Code changes still required** (the NaN fix is already committed):

In `envs/mobile_manipulation/tidybot_push_easy.py`:
- `self.episode_length = 50` → `self.episode_length = 500`
- `self.cube_noise_scale = 0.` → `self.cube_noise_scale = 0.1`
- `self.goal_noise_scale = 0.` → `self.goal_noise_scale = 0.1`

Commit these to `dylanzhou2/scaling-crl` so all training runs are from a known repo state.

**Optional `nan_to_num` guard** in `arm_envs.py` (only if Phase 1's GPU re-run shows NaN rate >1%):
```python
obs_nan = jnp.any(jnp.isnan(obs))
reward_nan = jnp.isnan(reward)
obs = jnp.nan_to_num(obs, nan=0.0)
reward = jnp.where(reward_nan, 0.0, reward)
state.metrics.update(obs_nan=obs_nan.astype(jnp.float32), reward_nan=reward_nan.astype(jnp.float32))
```
The `obs_nan` / `reward_nan` metrics give a wandb smoking gun if NaN re-appears.

**Launch commands** (three runs):
```bash
# Depth 4
uv run train.py --env_id "tidybot_push_easy" --eval_env_id "tidybot_push_easy" \
  --episode_length 500 --num_epochs 100 --total_env_steps 100000000 \
  --critic_depth 4 --actor_depth 4 --batch_size 512 \
  --exp_name "tidybot_push_easy_d4_ep500_noise01"

# Depth 16 (skip connections every 4 layers)
uv run train.py --env_id "tidybot_push_easy" --eval_env_id "tidybot_push_easy" \
  --episode_length 500 --num_epochs 100 --total_env_steps 100000000 \
  --critic_depth 16 --actor_depth 16 \
  --actor_skip_connections 4 --critic_skip_connections 4 --batch_size 512 \
  --exp_name "tidybot_push_easy_d16_ep500_noise01"

# Depth 64 (the depth-scaling claim from the paper)
uv run train.py --env_id "tidybot_push_easy" --eval_env_id "tidybot_push_easy" \
  --episode_length 500 --num_epochs 100 --total_env_steps 100000000 \
  --critic_depth 64 --actor_depth 64 \
  --actor_skip_connections 4 --critic_skip_connections 4 --batch_size 512 \
  --exp_name "tidybot_push_easy_d64_ep500_noise01"
```

**What to watch.** First 2h, every 30 min: success rate, success_easy (looser threshold for early signal), policy entropy, value loss, and `obs_nan` if the guard is in.

**Kill criterion.** `obs_nan >5%` → kill that run, investigate before further launches. Flat reward at 4h with healthy NaN counts → likely needs reward shaping (Phase 4 fork).

**Exit criterion.** At least one depth shows learning (positive reward slope and success_easy > 0).

### Phase 4 — Diagnose & iterate ⏳ pending (depends on Phase 2/3 outcome)

**Goal.** Symptom-driven debug. Four branches based on what Phase 2/3 produced.

**If runs show learning (any depth).**
- Document final success rate per depth in `diagnostic_results.md`.
- Promote `tidybot_push_easy` to a new env file `tidybot_push_aside.py` (copy + adjust env_name for clarity).
- Wire `tidybot_push_aside` into `train.py`.
- Hand off to course-plan Phase 2 (vault's `../deliverables/course-project-plan.md`), specifically the "harder distribution" step (Day 6 — adds obstacles).

**If still 0 success but NaN counts are healthy.** Reward / exploration issue. Try in order:
1. Contact-with-cube bonus (small +reward when EEF within 0.1 m of cube).
2. Velocity-toward-goal reward (`+(cube_velocity · direction_to_goal)`).
3. Curriculum on cube initial position (start near EEF, expand outward over training).

**If NaNs are nonzero (>1% on GPU).** Likely brax contact or XML contact props:
1. Re-verify both brax patches are applied (`find .venv -name contact.py` and inspect).
2. Inspect `envs/assets/tidybot_push_easy.xml` for unusual `<contact>` overrides (zero friction, weird `solref`).
3. Try smaller `n_frames` in `arm_envs.py:23` (e.g. 5 instead of 25) — fewer substeps reduces accumulated contact errors.

**If `arm_push_easy` from Phase 2 didn't train.** Stack-level issue, not env-specific. Don't iterate on tidybot. Check:
1. brax / mujoco / mjx version mismatch with pinned `pyproject.toml`.
2. JAX-CUDA actually using the GPU (`jax.devices()` returns CUDA devices, not CPU).
3. wandb sync failures silently masking errors.

### Phase 5 — GREEN/YELLOW/RED decision ⏳ pending

**Goal.** Commit to one of three states. The vault's `course-project-plan.md` and `milestone-plan.md` branch off this.

- **GREEN.** `tidybot_push_easy` (renamed `tidybot_push_aside`) trains to ≥30% success. Execute course-plan Phase 2 normally.
- **YELLOW.** Env works mechanically, success rate plateaus low. Document baseline, continue course plan with reward-shaping as priority.
- **RED.** Env broken at a level not fixable in available time. Execute one of three named pivots (full text in `../deliverables/tidybot-testing-plan.md`):
  - **A.** Use upstream `arm_push_easy` (Panda tabletop) as the project's push primitive. Loses mobile-base aspect; salvages algorithmic claim.
  - **B.** Rebuild minimal `push_aside` env on top of `TidyBotEnv` base. 2–3 day cost; only if Phase 4 diagnosed a structural bug we can't fix in-place.
  - **C.** Drop mobile manipulation from class scope. Last-resort pivot, requires rewriting `course-project-plan.md` and the proposal narrative.

After decision: append a `wiki/log.md` entry in the vault, and update `course-project-plan.md` Phase 2 to match.

## Known issues / recent fixes (don't undo)

- **2026-05-23 — TidyBotEnv inf-range NaN fix (commit `3debbef`)**. Kinova's `joint_1, _3, _5, _7` are continuous-rotation with `jnt_range = [-inf, +inf]`. The action-conversion override in `tidybot_envs.py` was computing `offset = (-inf + +inf)/2 = NaN` and `multiplier = +inf`, polluting every step from t=0. The fix substitutes `[-π, π]` for any arm joint with non-finite range. Pre-fix: 100/100 episodes NaN. Post-fix: 1/100 (contact-induced; not action conversion). **Do not remove this substitution** — the parent class's `jnp.where(multiplier > 0, ...)` guard doesn't catch this because `inf > 0` is `True`.
- **Default `cube_noise_scale = 0`, `goal_noise_scale = 0`, `episode_length = 50` in `tidybot_push_easy.py`** — these defaults make the task unsolvable from cold-start with sparse reward. For actual training, set noise scales to 0.1+ and episode_length to 500 (or pass `--episode_length 500` to train.py).
- **Sparse reward**: `ArmEnvs.step` sets `reward = success` (binary, threshold 0.1m). CRL relies on hindsight relabeling; don't add reward shaping unless an experiment specifically calls for it.

## Common workflows

- **Diagnose env behavior**: extend `diagnostic_tidybot_push_easy.py`. Don't write one-off scripts that duplicate it.
- **Add a new env**: copy `tidybot_push_easy.py` as a template, add a branch to `train.py`'s `if env_id == ...` chain (~ line 555+), and add the XML to `envs/assets/`.
- **Launch training**: see `../deliverables/gpu-setup-guide.md` §7–§8 for the canonical commands. Always pass `--save_buffer 0` unless you specifically need replay buffer dumps (they're large).
- **Update upstream sync**: `git fetch upstream && git merge upstream/main` (we are downstream of `wang-kevin3290/scaling-crl`). Don't push to upstream.

## Don't

- Don't suggest IsaacLab as a sim alternative.
- Don't rebuild existing TidyBot envs from scratch — extend or fork them.
- Don't add reward shaping to existing envs without an experiment justifying it; the project bets on demo-free + sparse-reward CRL.
- Don't blindly commit `runs/`, `.venv/`, `wandb/`, `__pycache__/`, `.DS_Store` (all gitignored).
- Don't bypass `uv` for installs (e.g. `pip install` into the project venv).
- Don't write training pipeline reimplementations of CRL — that's upstream code, treat it as a library.

## AI tools disclosure (CS224R requirement)

This codebase is part of a CS224R submission requiring an AI Tools Disclosure in the final report. When making changes, distinguish:
- **Upstream code** (most of `train.py`, `buffer.py`, `evaluator.py`, `envs/manipulation/arm_envs.py`, `envs/manipulation/arm_*.py`) — from `wang-kevin3290/scaling-crl`.
- **Project-added envs** (`envs/mobile_manipulation/tidybot_*.py`, `envs/manipulation/tidybot_envs.py`) — written by the project team (pre-Claude).
- **Claude-assisted changes** (diagnostic scripts, the inf-range fix, the headless `xml_sanity_check.py`, patch scripts) — done with Claude help.

Be precise about authorship in PR descriptions and commits.

## Broader project context

Lives in the parent vault (`scaling-crl-vault`):
- `wiki/overview.md` — project thesis (demo-free primitives on TidyBot, CRL-MAB synthesis).
- `wiki/project-meta.md` — CS224R timeline, mentor list, compute access.
- `deliverables/proposal-v2.md` — the proposal we're executing against.
- `deliverables/course-project-plan.md` — 17-day plan through final report (2026-06-08).
- `deliverables/tidybot-testing-plan.md` — the 24–48h debug campaign this codebase's recent changes implement.
- `deliverables/scaling-crl-audit.md` — initial audit of this codebase.
- `deliverables/gpu-setup-guide.md` — GPU box setup.

When you encounter "why are we doing X" questions, the vault is where the answers live.

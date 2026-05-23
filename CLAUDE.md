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

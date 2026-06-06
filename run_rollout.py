"""Render an HTML rollout of a trained policy.

Two modes:

  * Plain CRL checkpoint (`*_final.pkl` = (alpha, actor, critic)):
      uv run run_rollout.py --checkpoint_path .../...final.pkl --env_id arm_push_hard

  * MAB residual checkpoint (`residual_*.pkl` from train_residual_mab.py):
      uv run run_rollout.py --residual_path weights/.../residual_tidybot_push_easy_*.pkl

For a residual checkpoint the base CRL checkpoint, env_id, architecture, and the
barrier hyperparameters (epsilon / sigma_floor / barrier_type) are all read from
the residual pkl's `meta`, so only `--residual_path` is required. Use
`--no_residual` to render the frozen base policy instead of the corrected one
(handy for a side-by-side base-vs-corrected comparison).

Cross-object transfer: because arm_push_{hard,sphere,bar} share the same obs
(20-dim) and action (5-dim) spaces, a policy trained on one object can be rendered
on another by overriding --env_id. E.g. an arm_push_hard (cube) checkpoint on the
bar env:
    uv run run_rollout.py --checkpoint_path .../arm_push_hard..._final.pkl \\
        --env_id arm_push_bar --out rollout_bar.html

The env_id is resolved by make_env() in train_residual_mab.py; supported ids
include arm_push_{easy,hard,sphere,bar} and tidybot_push_{easy,hard}.
"""

import argparse

import jax
import jax.numpy as jnp
from flax import linen as nn
from brax.io import html

from train import Actor, load_params
# Reuse the residual network class and env dispatch so the param trees and the
# barrier math match train_residual_mab.py bit-for-bit.
from train_residual_mab import make_env, ResidualActor


def parse_args():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument(
      "--residual_path", default="",
      help="Path to a residual_*.pkl from train_residual_mab.py. If set, renders "
           "the MAB-corrected policy (base checkpoint + arch read from its meta).")
  p.add_argument(
      "--checkpoint_path", default="",
      help="Path to a plain CRL *_final.pkl. Used when --residual_path is unset, "
           "or to override the base checkpoint named in the residual meta.")
  p.add_argument(
      "--env_id", default="",
      help="Env id. Required for a plain checkpoint; for a residual checkpoint it "
           "defaults to the env_id stored in the meta.")
  p.add_argument(
      "--no_residual", action="store_true",
      help="With --residual_path, render the frozen BASE policy (residual off) "
           "for a base-vs-corrected comparison.")
  p.add_argument("--steps", type=int, default=1000, help="Max rollout steps.")
  p.add_argument("--seed", type=int, default=0, help="Reset PRNG seed.")
  p.add_argument("--out", default="rollout.html", help="Output HTML path.")
  p.add_argument(
      "--freeze_base", action="store_true",
      help="Zero the 3 base action dims each step so the mobile base holds its "
           "position (base_action = base_current + 0). Tidybot envs only; useful "
           "for inspecting arm-only behavior or matching a base-frozen training run.")
  # arm_push_aside knobs. Default None -> use the residual meta if present, else
  # the env's own defaults. Provide explicitly to render a plain checkpoint on a
  # specific OOD offset.
  p.add_argument("--aside_offset", type=float, default=None)
  p.add_argument("--zone_radius", type=float, default=None)
  p.add_argument("--aside_axis", type=int, default=None)
  return p.parse_args()


def main():
  args = parse_args()
  if not args.residual_path and not args.checkpoint_path:
    raise ValueError("Provide --residual_path or --checkpoint_path.")

  # ------------------------------------------------------------------
  # Resolve config (residual meta is the source of truth when present)
  # ------------------------------------------------------------------
  meta = None
  res_params = None
  if args.residual_path:
    blob = load_params(args.residual_path)
    res_params = blob["residual_params"]
    meta = blob["meta"]
    cfg = meta["arch"]
    env_id = args.env_id or meta["env_id"]
    base_ckpt = args.checkpoint_path or meta["checkpoint_path"]
    use_residual = not args.no_residual
    print(f"[load] residual params from {args.residual_path}")
    print(f"[meta] env_id={meta['env_id']} epsilon={meta['epsilon']} "
          f"sigma_floor={meta['sigma_floor']} barrier_type={meta['barrier_type']}")
  else:
    # Plain CRL checkpoint: read arch from its sibling *_args.pkl.
    args_path = args.checkpoint_path.replace("_final.pkl", "_args.pkl")
    saved = load_params(args_path)
    cfg = dict(
        actor_network_width=saved.actor_network_width,
        actor_depth=saved.actor_depth,
        actor_skip_connections=saved.actor_skip_connections,
        use_relu=saved.use_relu,
    )
    env_id = args.env_id
    base_ckpt = args.checkpoint_path
    use_residual = False
    if not env_id:
      raise ValueError("--env_id is required for a plain checkpoint.")

  # ------------------------------------------------------------------
  # Environment (unwrapped, so pipeline_state is available for rendering)
  # ------------------------------------------------------------------
  # Forward arm_push_aside knobs: CLI override > residual meta > env default.
  def _pick(cli_val, key, default):
    if cli_val is not None:
      return cli_val
    if meta is not None and key in meta:
      return meta[key]
    return default

  env_kwargs = {}
  if env_id == "arm_push_aside":
    env_kwargs = dict(
        aside_offset=_pick(args.aside_offset, "aside_offset", 0.15),
        zone_radius=_pick(args.zone_radius, "zone_radius", 0.12),
        aside_axis=_pick(args.aside_axis, "aside_axis", 0),
    )

  env = make_env(env_id, **env_kwargs)
  action_size = env.action_size
  obs_size = env.observation_size
  print(f"[env] {env_id} obs_size={obs_size} action_size={action_size}")

  # ------------------------------------------------------------------
  # Frozen base actor (+ optional residual)
  # ------------------------------------------------------------------
  actor = Actor(
      action_size=action_size,
      network_width=cfg["actor_network_width"],
      network_depth=cfg["actor_depth"],
      skip_connections=cfg["actor_skip_connections"],
      use_relu=cfg["use_relu"],
  )
  _alpha, actor_params, _critic = load_params(base_ckpt)
  print(f"[load] frozen base actor from {base_ckpt}")

  if use_residual:
    residual = ResidualActor(
        action_size=action_size,
        network_width=meta["residual_width"],
        network_depth=meta["residual_depth"],
        use_relu=cfg["use_relu"],
    )
    eps = meta["epsilon"]
    sigma_floor = meta["sigma_floor"]
    ball = meta["barrier_type"] == "ball"

    @jax.jit
    def policy(obs):
      mean, log_std = actor.apply(actor_params, obs)
      std = jnp.exp(log_std)
      sigma_eff = jnp.maximum(std, sigma_floor)
      raw = residual.apply(res_params, obs)
      if ball:
        norm = jnp.sqrt(jnp.sum(raw ** 2, axis=-1, keepdims=True) + 1e-8)
        u = eps * raw / jnp.maximum(1.0, norm)  # project into L2 ball, radius eps
      else:
        u = eps * jnp.tanh(raw)  # per-dim box of half-width eps
      dx = sigma_eff * u  # de-whiten into pre-tanh action coords
      return jnp.tanh(mean + dx)

    print(f"[policy] MAB-corrected (epsilon={eps}, barrier={meta['barrier_type']})")
  else:
    @jax.jit
    def policy(obs):
      mean, _ = actor.apply(actor_params, obs)
      return nn.tanh(mean)

    print("[policy] frozen base (residual off)" if args.residual_path
          else "[policy] frozen base")

  # ------------------------------------------------------------------
  # Rollout
  # ------------------------------------------------------------------
  env_step = jax.jit(env.step)
  env_reset = jax.jit(env.reset)

  env_state = env_reset(jax.random.PRNGKey(args.seed))
  rollout_states = []
  total_reward = 0.0

  if args.freeze_base:
    print("[base] FROZEN — base action dims (0,1,2) zeroed each step.")

  for _ in range(args.steps):
    obs = jnp.expand_dims(env_state.obs, axis=0)
    action = policy(obs)[0]
    if args.freeze_base:
      action = action.at[:3].set(0.0)
    env_state = env_step(env_state, action)
    rollout_states.append(env_state.pipeline_state)
    total_reward += env_state.reward.item()
    if env_state.done:
      break

  print(f"Episode reward: {total_reward:.4f} over {len(rollout_states)} steps")

  # ------------------------------------------------------------------
  # Render
  # ------------------------------------------------------------------
  html_string = html.render(env.sys, rollout_states)
  with open(args.out, "w") as f:
    f.write(html_string)
  print(f"Saved rollout to {args.out}")


if __name__ == "__main__":
  main()

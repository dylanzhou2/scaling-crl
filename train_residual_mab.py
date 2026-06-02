"""Train a Mahalanobis Action Barrier (MAB) residual on top of a frozen CRL policy.

This does NOT retrain CRL. It loads a finished `final.pkl` checkpoint
(`(alpha_params, actor_params, critic_params)`), freezes the actor and the
contrastive critic, and trains a small residual network `r(s)` whose output is
constrained to a Mahalanobis ball defined by the policy's own learned action
covariance. The corrected action is

    a = tanh( mean(s) + dx ),   dx = sigma_eff(s) * u,   ||u|| <= epsilon

where (mean, log_std) come from the frozen actor (so sigma = exp(log_std) is the
state-dependent action std, i.e. a diagonal Mahalanobis metric), sigma_eff floors
sigma so the barrier doesn't vanish when the policy gets confident, and u is the
barrier-constrained residual. The residual is trained to maximize the frozen
contrastive Q = -||sa_repr(s, a) - g_repr(g)|| toward the *commanded* goal carried
in the observation tail.

Author: Claude-assisted (MAB residual stage). The frozen actor/critic and the
network class definitions are upstream / project code, imported from train.py.

Usage (arch is auto-read from the checkpoint's sibling *_args.pkl):

  uv run train_residual_mab.py \
    --checkpoint_path weights/tidybot_push_easy/scaling-crl_runs_tidybot_push_easy_1000_20260601-190543_final.pkl \
    --env_id tidybot_push_easy \
    --exp_name mab_d4_eps0.3

Ablations (--barrier_type, mirrors the project's CRL-MAB standard ablation ladder):

  none                    # unconstrained residual; tests whether residuals can
                          # hack the frozen contrastive critic when no barrier
                          # is imposed at all.
  box                     # per-dim tanh box (LATENT-style lambda*sigma*tanh(z))
                          # in the actor's per-dim log_std metric.
  ball  (default)         # diagonal Mahalanobis: L2 ball of radius eps in
                          # whitened (sigma_eff) coords. Sigma = actor's per-dim
                          # learned action std. This is the action-space MAB.
  euclidean               # L2 ball of radius eps in raw action space (Sigma = I).
                          # Isolates the *Mahalanobis* effect from the "constrained
                          # at all" effect by stripping per-dim sigma scaling.
  isotropic               # L2 ball with SCALAR sigma = mean(sigma_eff) over dims.
                          # Isolates whether per-dim sigma weighting matters
                          # beyond just having an adaptive overall scale.
  embedding_ball          # ball (action space) + Lagrangian soft penalty on
                          # ||phi(s, a_corrected) - phi(s, a_base)||_2 > eps_phi.
                          # Embedding-space barrier with Sigma_phi = I.
  embedding_mahalanobis   # ball (action space) + Lagrangian soft penalty on
                          # whitened ||delta phi||_{Sigma_phi^-1}, with diagonal
                          # Sigma_phi estimated from the current minibatch's
                          # base-action embeddings. Embedding-space MAB (the
                          # theory-justified variant per the project's theory plan).

Other ablation knobs:
  --epsilon 0.0            # barrier shut -> corrected == base (sanity / baseline)
  --sigma_floor 0.0        # pure policy-covariance metric, no floor on sigma
  --embedding_epsilon X    # radius of soft-constraint ball in critic embedding space
  --embedding_penalty X    # Lagrangian coefficient for embedding-space penalty
"""

from dataclasses import dataclass
from datetime import datetime
import os
import pickle
from pathlib import Path
from typing import Optional

from brax import envs
import flax.linen as nn
from flax.linen.initializers import variance_scaling
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

# Reuse the EXACT network definitions and IO helpers from the training script so
# the loaded param trees match bit-for-bit. Importing train.py is safe: its main
# body is guarded by `if __name__ == "__main__"`.
from train import (
    Actor,
    SA_encoder,
    G_encoder,
    residual_block,
    load_params,
    save_params,
    Args,  # noqa: F401  (kept importable so pickled args.pkl resolves)
)

# The saved `*_args.pkl` files were pickled while train.py ran as `__main__`, so
# their class path is `__main__.Args`. Bind that name on whatever module is
# currently `__main__` so the unpickler can resolve it (works both when this file
# is run directly and when it is imported elsewhere).
import sys as _sys
try:
  setattr(_sys.modules["__main__"], "Args", Args)
except Exception:  # noqa: BLE001
  pass


@dataclass
class ResArgs:
  # --- what to load / where it came from ---
  checkpoint_path: str = ""
  """Path to a CRL `*_final.pkl` = (alpha_params, actor_params, critic_params)."""
  args_path: str = ""
  """Path to the sibling `*_args.pkl`. Default: derive from checkpoint_path."""
  env_id: str = "tidybot_push_easy"

  # --- architecture overrides (None => read from the saved args.pkl) ---
  obs_dim: Optional[int] = None
  goal_start_idx: Optional[int] = None
  goal_end_idx: Optional[int] = None
  actor_network_width: Optional[int] = None
  actor_depth: Optional[int] = None
  actor_skip_connections: Optional[int] = None
  critic_network_width: Optional[int] = None
  critic_depth: Optional[int] = None
  critic_skip_connections: Optional[int] = None
  use_relu: Optional[int] = None
  episode_length: Optional[int] = None

  # --- arm_push_aside env knobs (the OOD-offset sweep variable) ---
  aside_offset: float = 0.15
  """Rightward goal shift along the lateral axis: goal_x = 0.25 + aside_offset.
  In-distribution for offset <= 0.25 (red bin edge); OOD beyond. Only used by
  --env_id arm_push_aside."""
  zone_radius: float = 0.12
  """Region success radius for arm_push_aside (looser than the 0.1 m coordinate bar)."""
  aside_axis: int = 0
  """Lateral axis for the aside offset (0=x/left-right, 1=y/depth)."""

  # --- MAB residual hyperparameters ---
  epsilon: float = 0.3
  """Mahalanobis ball radius (in whitened pre-tanh units). 0 => no correction."""
  sigma_floor: float = 0.1
  """Floor on per-dim action std used in the metric, so the barrier keeps
  authority when the policy's std collapses late in training."""
  barrier_type: str = "ball"
  """One of: 'none' (unconstrained), 'box' (per-dim tanh), 'ball' (default,
  action-space diagonal Mahalanobis), 'euclidean' (action-space L2 with Sigma=I),
  'isotropic' (action-space L2 with scalar sigma), 'embedding_ball' (ball +
  soft phi-space L2 penalty), 'embedding_mahalanobis' (ball + soft phi-space
  diagonal Mahalanobis penalty). See top-of-file docstring for full semantics."""
  embedding_epsilon: float = 1.0
  """Radius (in phi-space units) of the embedding-space soft constraint.
  Used only for barrier_type in {'embedding_ball', 'embedding_mahalanobis'}."""
  embedding_penalty: float = 1.0
  """Lagrangian coefficient for the embedding-space hinge penalty.
  loss += embedding_penalty * mean( max(0, ||delta phi||^2 - embedding_epsilon^2) )."""
  residual_width: int = 256
  residual_depth: int = 4
  residual_lr: float = 3e-4
  residual_reg: float = 1e-3
  """Small L2 penalty on the whitened residual u, to prefer minimal corrections."""

  # --- training loop ---
  seed: int = 0
  num_envs: int = 256
  rollout_steps_per_iter: int = 100
  num_iterations: int = 20
  grad_steps_per_iter: int = 200
  batch_size: int = 512
  collect_stochastic: int = 1
  """1 => add base-policy exploration noise while collecting states for coverage."""

  # --- eval / io ---
  num_eval_envs: int = 128
  eval_every: int = 2
  success_threshold: float = 0.5
  exp_name: str = "mab_residual"
  out_dir: str = ""
  """Where to save the residual params. Default: alongside the checkpoint."""


class ResidualActor(nn.Module):
  """Small MLP producing a raw (pre-barrier) action-space residual from obs.

  Mirrors the body style of train.Actor (LayerNorm + swish + residual blocks).
  The final layer is zero-initialized so the residual starts at exactly 0 and
  the corrected policy begins identical to the frozen base policy.
  """

  action_size: int
  network_width: int = 256
  network_depth: int = 4
  use_relu: int = 0

  @nn.compact
  def __call__(self, x):
    lecun = variance_scaling(1 / 3, "fan_in", "uniform")
    bias_init = nn.initializers.zeros
    normalize = lambda y: nn.LayerNorm()(y)
    activation = nn.relu if self.use_relu else nn.swish

    x = nn.Dense(self.network_width, kernel_init=lecun, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    for _ in range(self.network_depth // 4):
      x = residual_block(x, self.network_width, normalize, activation)
    # Zero-init head => raw residual == 0 at start => corrected == base policy.
    raw = nn.Dense(
        self.action_size,
        kernel_init=nn.initializers.zeros,
        bias_init=nn.initializers.zeros,
    )(x)
    return raw


def make_env(env_id, **env_kwargs):
  """Construct the (unwrapped) brax env. Mirrors train.py's dispatch for the
  envs this residual stage targets; extend here if you add more.

  env_kwargs are forwarded only to envs that accept them (currently
  arm_push_aside); other branches ignore them.
  """
  if env_id == "tidybot_push_easy":
    from envs.mobile_manipulation.tidybot_push_easy import TidyBotPushEasy

    return TidyBotPushEasy(backend="mjx")
  elif env_id == "tidybot_push_hard":
    from envs.mobile_manipulation.tidybot_push_hard import TidyBotPushHard

    return TidyBotPushHard(backend="mjx")
  elif env_id == "arm_push_easy":
    from envs.manipulation.arm_push_easy import ArmPushEasy

    return ArmPushEasy(backend="mjx")
  elif env_id == "arm_push_hard":
    from envs.manipulation.arm_push_hard import ArmPushHard

    return ArmPushHard(backend="mjx")
  elif env_id == "arm_push_sphere":
    from envs.manipulation.arm_push_sphere import ArmPushSphere

    return ArmPushSphere(backend="mjx")
  elif env_id == "arm_push_bar":
    from envs.manipulation.arm_push_bar import ArmPushBar

    return ArmPushBar(backend="mjx")
  elif env_id == "arm_push_shifted":
    from envs.manipulation.arm_push_shifted import ArmPushShifted

    return ArmPushShifted(backend="mjx")
  elif env_id == "arm_push_aside":
    from envs.manipulation.arm_push_aside import ArmPushAside

    return ArmPushAside(
        backend="mjx",
        aside_offset=env_kwargs.get("aside_offset", 0.15),
        zone_radius=env_kwargs.get("zone_radius", 0.12),
        aside_axis=env_kwargs.get("aside_axis", 0),
    )
  else:
    raise NotImplementedError(
        f"make_env: env_id '{env_id}' not wired into train_residual_mab.py. "
        "Add a branch here mirroring train.py."
    )


def resolve_arch(args):
  """Fill architecture/obs fields from the saved args.pkl, with CLI overrides."""
  saved = None
  args_path = args.args_path
  if not args_path and args.checkpoint_path:
    # `..._final.pkl` -> `..._args.pkl`
    args_path = args.checkpoint_path.replace("_final.pkl", "_args.pkl")
  if args_path and os.path.exists(args_path):
    try:
      saved = load_params(args_path)  # a pickled train.Args instance
      print(f"[arch] loaded saved args from {args_path}", flush=True)
    except Exception as e:  # noqa: BLE001
      print(f"[arch] could not unpickle {args_path} ({e}); using CLI/defaults",
            flush=True)

  def pick(field, default):
    cli_val = getattr(args, field)
    if cli_val is not None:
      return cli_val
    if saved is not None and hasattr(saved, field):
      return getattr(saved, field)
    return default

  # Last-resort obs/goal defaults (used only if no args.pkl and no CLI value),
  # keyed by env family to match train.py's dispatch.
  if args.env_id.startswith("arm_"):
    obs_d, gstart, gend, ep = 17, 0, 3, 1000
  else:  # tidybot_push_*
    obs_d, gstart, gend, ep = 20, 10, 13, 500

  cfg = dict(
      obs_dim=pick("obs_dim", obs_d),
      goal_start_idx=pick("goal_start_idx", gstart),
      goal_end_idx=pick("goal_end_idx", gend),
      actor_network_width=pick("actor_network_width", 256),
      actor_depth=pick("actor_depth", 4),
      actor_skip_connections=pick("actor_skip_connections", 0),
      critic_network_width=pick("critic_network_width", 256),
      critic_depth=pick("critic_depth", 4),
      critic_skip_connections=pick("critic_skip_connections", 0),
      use_relu=pick("use_relu", 0),
      episode_length=pick("episode_length", ep),
  )
  return cfg


def main(args):
  print("JAX devices:", jax.devices(), flush=True)
  if not args.checkpoint_path:
    raise ValueError("--checkpoint_path is required (a CRL *_final.pkl).")

  cfg = resolve_arch(args)
  obs_dim = cfg["obs_dim"]
  goal_start, goal_end = cfg["goal_start_idx"], cfg["goal_end_idx"]
  print(f"[arch] {cfg}", flush=True)

  # ---- env ----
  env = make_env(
      args.env_id,
      aside_offset=args.aside_offset,
      zone_radius=args.zone_radius,
      aside_axis=args.aside_axis,
  )
  env = envs.training.wrap(env, episode_length=cfg["episode_length"])
  env.step = jax.jit(env.step)
  reset = jax.jit(env.reset)
  obs_size = env.observation_size
  action_size = env.action_size
  print(f"[env] obs_size={obs_size} action_size={action_size}", flush=True)
  assert obs_size - obs_dim == goal_end - goal_start, (
      f"obs tail ({obs_size - obs_dim}) != goal dim ({goal_end - goal_start}); "
      "check obs_dim / goal indices."
  )

  # ---- frozen networks (architecture must match the checkpoint) ----
  actor = Actor(
      action_size=action_size,
      network_width=cfg["actor_network_width"],
      network_depth=cfg["actor_depth"],
      skip_connections=cfg["actor_skip_connections"],
      use_relu=cfg["use_relu"],
  )
  sa_encoder = SA_encoder(
      network_width=cfg["critic_network_width"],
      network_depth=cfg["critic_depth"],
      skip_connections=cfg["critic_skip_connections"],
      use_relu=cfg["use_relu"],
  )
  g_encoder = G_encoder(
      network_width=cfg["critic_network_width"],
      network_depth=cfg["critic_depth"],
      skip_connections=cfg["critic_skip_connections"],
      use_relu=cfg["use_relu"],
  )

  loaded = load_params(args.checkpoint_path)
  if not (isinstance(loaded, (tuple, list)) and len(loaded) == 3):
    raise ValueError(
        "Expected checkpoint to be (alpha_params, actor_params, critic_params); "
        f"got {type(loaded)} of len "
        f"{len(loaded) if hasattr(loaded, '__len__') else '?'}."
    )
  _alpha_params, actor_params, critic_params = loaded
  if "sa_encoder" not in critic_params or "g_encoder" not in critic_params:
    raise ValueError(
        "critic_params missing 'sa_encoder'/'g_encoder' -- this checkpoint is "
        "not a CRL critic (SAC/VPG checkpoints are unsupported here)."
    )
  print(f"[load] frozen actor + contrastive critic from {args.checkpoint_path}",
        flush=True)

  # ---- residual network ----
  residual = ResidualActor(
      action_size=action_size,
      network_width=args.residual_width,
      network_depth=args.residual_depth,
      use_relu=cfg["use_relu"],
  )
  key = jax.random.PRNGKey(args.seed)
  key, rinit = jax.random.split(key)
  res_params = residual.init(rinit, np.ones([1, obs_size]))
  res_state = TrainState.create(
      apply_fn=residual.apply,
      params=res_params,
      tx=optax.chain(
          optax.clip_by_global_norm(1.0),
          optax.adam(learning_rate=args.residual_lr),
      ),
  )

  eps = args.epsilon
  sigma_floor = args.sigma_floor
  barrier_type = args.barrier_type
  valid_barrier_types = (
      "none", "box", "ball", "euclidean", "isotropic",
      "embedding_ball", "embedding_mahalanobis",
  )
  if barrier_type not in valid_barrier_types:
    raise ValueError(
        f"--barrier_type={barrier_type!r} not recognised; "
        f"choose from {valid_barrier_types}"
    )
  embedding_eps = args.embedding_epsilon
  embedding_pen = args.embedding_penalty
  emb_variant = barrier_type.startswith("embedding")

  def barrier_residual(res_params, obs):
    """Compute (mean, std, dx, u) for the chosen barrier_type.

    Semantics of dx (added to mean pre-tanh) and u (whitened residual; the L2
    reg term penalises ||u||^2):
      none         : dx = raw,           u = raw       (unconstrained)
      box          : dx = sigma_eff * u, u = eps * tanh(raw)             (per-dim box)
      ball         : dx = sigma_eff * u, u = eps * raw / max(1, ||raw||) (diag Mahalanobis)
      euclidean    : dx = u,             u = eps * raw / max(1, ||raw||) (Sigma = I)
      isotropic    : dx = sigma_s * u,   u = eps * raw / max(1, ||raw||) (scalar sigma_s)
      embedding_*  : action-space dx as in 'ball'; constraint enforced softly
                     in the loss via a hinge on ||delta phi||.
    """
    mean, log_std = actor.apply(actor_params, obs)  # frozen base
    std = jnp.exp(log_std)
    sigma_eff = jnp.maximum(std, sigma_floor)
    raw = residual.apply(res_params, obs)

    if barrier_type == "none":
      dx = raw
      u = raw
    elif barrier_type == "box":
      u = eps * jnp.tanh(raw)
      dx = sigma_eff * u
    else:
      # All ball-like barriers share the same whitened L2 projection of `raw`.
      norm = jnp.sqrt(jnp.sum(raw ** 2, axis=-1, keepdims=True) + 1e-8)
      u = eps * raw / jnp.maximum(1.0, norm)
      if barrier_type == "euclidean":
        dx = u                                       # Sigma = I (action space)
      elif barrier_type == "isotropic":
        sigma_scalar = jnp.mean(sigma_eff, axis=-1, keepdims=True)
        dx = sigma_scalar * u                        # scalar per-state sigma
      else:
        # 'ball' (diagonal Mahalanobis) or 'embedding_*' (action-space part).
        dx = sigma_eff * u
    return mean, std, dx, u

  def corrected_det(res_params, obs):
    mean, _std, dx, _u = barrier_residual(res_params, obs)
    return jnp.tanh(mean + dx)

  # ---- residual loss: maximize frozen contrastive Q toward the commanded goal ----
  @jax.jit
  def update(res_state, obs_batch):
    def loss_fn(rp):
      mean, _std, dx, u = barrier_residual(rp, obs_batch)
      action = jnp.tanh(mean + dx)
      state = obs_batch[:, :obs_dim]
      goal = obs_batch[:, obs_dim:]
      sa_repr = sa_encoder.apply(critic_params["sa_encoder"], state, action)
      g_repr = g_encoder.apply(critic_params["g_encoder"], goal)
      q = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1) + 1e-6)
      loss = -jnp.mean(q) + args.residual_reg * jnp.mean(jnp.sum(u ** 2, axis=-1))

      # Embedding-space soft constraint for embedding_* barrier types.
      # phi(s, .) has no dependence on res_params via base_action (it uses the
      # frozen actor mean), so the gradient flows only through sa_repr.
      if emb_variant:
        base_action = jnp.tanh(mean)
        sa_base = sa_encoder.apply(
            critic_params["sa_encoder"], state, base_action
        )
        delta_phi = sa_repr - sa_base
        if barrier_type == "embedding_mahalanobis":
          # Diagonal Sigma_phi estimated from this batch's base embeddings.
          phi_centered = sa_base - jnp.mean(sa_base, axis=0, keepdims=True)
          phi_var = jnp.mean(phi_centered ** 2, axis=0, keepdims=True) + 1e-6
          delta_phi_eff = delta_phi / jnp.sqrt(phi_var)
        else:  # embedding_ball: Sigma_phi = I
          delta_phi_eff = delta_phi
        phi_dist_sq = jnp.sum(delta_phi_eff ** 2, axis=-1)
        # Hinge: only penalise when ||delta phi||^2 exceeds embedding_eps^2.
        phi_violation = jnp.maximum(phi_dist_sq - embedding_eps ** 2, 0.0)
        loss = loss + embedding_pen * jnp.mean(phi_violation)
        phi_dist = jnp.mean(jnp.sqrt(phi_dist_sq + 1e-8))
      else:
        phi_dist = jnp.asarray(0.0, dtype=dx.dtype)

      dx_norm = jnp.mean(jnp.sqrt(jnp.sum(dx ** 2, axis=-1) + 1e-8))
      return loss, (jnp.mean(q), dx_norm, phi_dist)

    (loss, (q, dx_norm, phi_dist)), grad = jax.value_and_grad(
        loss_fn, has_aux=True
    )(res_state.params)
    res_state = res_state.apply_gradients(grads=grad)
    return res_state, loss, q, dx_norm, phi_dist

  # ---- state collection: roll out the corrected policy (with base exploration) ----
  explore = bool(args.collect_stochastic)

  @jax.jit
  def collect(res_params, env_state, key):
    def step(carry, _):
      env_state, key = carry
      key, k = jax.random.split(key)
      mean, std, dx, _u = barrier_residual(res_params, env_state.obs)
      if explore:
        noise = jax.random.normal(k, shape=mean.shape, dtype=mean.dtype)
        action = jnp.tanh(mean + std * noise + dx)
      else:
        action = jnp.tanh(mean + dx)
      obs_t = env_state.obs
      env_state = env.step(env_state, action)
      return (env_state, key), obs_t

    (env_state, _), obs_stack = jax.lax.scan(
        step, (env_state, key), None, length=args.rollout_steps_per_iter
    )
    # (T, N, D) -> (T*N, D)
    obs_flat = obs_stack.reshape(-1, obs_stack.shape[-1])
    return env_state, obs_flat

  # ---- eval: deterministic rollout, base vs corrected ----
  @jax.jit
  def eval_rollout(res_params, env_state, use_residual):
    flag = jnp.float32(use_residual)

    def step(carry, _):
      env_state, sum_r, max_r = carry
      mean, _std, dx, _u = barrier_residual(res_params, env_state.obs)
      action = jnp.tanh(mean + flag * dx)
      env_state = env.step(env_state, action)
      r = env_state.reward
      return (env_state, sum_r + r, max_r * 0 + jnp.maximum(max_r, r)), None

    n = env_state.reward.shape[0]
    init = (env_state, jnp.zeros(n), jnp.zeros(n))
    (final, sum_r, max_r), _ = jax.lax.scan(
        step, init, None, length=cfg["episode_length"]
    )
    success = (max_r > args.success_threshold).astype(jnp.float32)
    return jnp.mean(sum_r), jnp.mean(success)

  def run_eval(tag, it):
    key_b = jax.random.PRNGKey(args.seed + 12345)
    key_c = jax.random.PRNGKey(args.seed + 12345)  # same resets => fair compare
    es_b = reset(jax.random.split(key_b, args.num_eval_envs))
    es_c = reset(jax.random.split(key_c, args.num_eval_envs))
    base_ret, base_succ = eval_rollout(res_state.params, es_b, 0)
    corr_ret, corr_succ = eval_rollout(res_state.params, es_c, 1)
    print(
        f"[eval it={it:>3} {tag}] "
        f"base: ret={float(base_ret):+.3f} succ={float(base_succ):.3f} | "
        f"corrected: ret={float(corr_ret):+.3f} succ={float(corr_succ):.3f}",
        flush=True,
    )
    return float(base_succ), float(corr_succ)

  # ---- training loop ----
  key, env_key = jax.random.split(key)
  env_state = reset(jax.random.split(env_key, args.num_envs))

  run_eval("init", 0)  # sanity: corrected should == base at it 0 (zero-init residual)

  for it in range(1, args.num_iterations + 1):
    key, ck = jax.random.split(key)
    env_state, obs_data = collect(res_state.params, env_state, ck)
    obs_data = jax.lax.stop_gradient(obs_data)
    m = obs_data.shape[0]

    last_loss = last_q = last_dx = last_phi = 0.0
    for _ in range(args.grad_steps_per_iter):
      key, ik = jax.random.split(key)
      idx = jax.random.randint(ik, (args.batch_size,), 0, m)
      obs_batch = obs_data[idx]
      res_state, loss, q, dx_norm, phi_dist = update(res_state, obs_batch)
      last_loss = float(loss)
      last_q = float(q)
      last_dx = float(dx_norm)
      last_phi = float(phi_dist)

    phi_str = f" |dphi|={last_phi:.4f}" if emb_variant else ""
    print(
        f"[train it={it:>3}] collected={m} loss={last_loss:+.4f} "
        f"Q={last_q:+.4f} |dx|={last_dx:.4f}{phi_str}",
        flush=True,
    )
    if args.eval_every and it % args.eval_every == 0:
      run_eval("", it)

  base_succ, corr_succ = run_eval("final", args.num_iterations)

  # ---- save ----
  out_dir = args.out_dir or os.path.dirname(args.checkpoint_path) or "."
  Path(out_dir).mkdir(parents=True, exist_ok=True)
  stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
  out_path = os.path.join(
      out_dir, f"residual_{args.env_id}_{args.exp_name}_{stamp}.pkl"
  )
  save_params(out_path, {
      "residual_params": res_state.params,
      "meta": {
          "env_id": args.env_id,
          "checkpoint_path": args.checkpoint_path,
          "aside_offset": args.aside_offset,
          "zone_radius": args.zone_radius,
          "aside_axis": args.aside_axis,
          "epsilon": args.epsilon,
          "sigma_floor": args.sigma_floor,
          "barrier_type": args.barrier_type,
          "embedding_epsilon": args.embedding_epsilon,
          "embedding_penalty": args.embedding_penalty,
          "residual_width": args.residual_width,
          "residual_depth": args.residual_depth,
          "arch": cfg,
          "final_base_success": base_succ,
          "final_corrected_success": corr_succ,
      },
  })
  print(f"[save] residual params -> {out_path}", flush=True)
  print(
      f"[done] base_succ={base_succ:.3f} corrected_succ={corr_succ:.3f} "
      f"delta={corr_succ - base_succ:+.3f}",
      flush=True,
  )
  return base_succ, corr_succ


if __name__ == "__main__":
  main(tyro.cli(ResArgs))

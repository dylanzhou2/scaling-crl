"""Constrained-exploration composition over frozen CRL primitives (LATENT-style).

This is NOT the MAB residual trainer (`train_residual_mab.py`), which refines ONE
frozen policy to better satisfy that policy's own frozen critic on the SAME task.
Here we LEARN a NEW task (the hallway) by composing several frozen CRL primitives
(navigate, push_aside, ...), with exploration CONSTRAINED to the primitives'
Mahalanobis balls so the composite policy can only take motions the primitives
already know. The objective is the hallway's own sparse TASK reward, optimized by
policy gradient (REINFORCE) -- there is no CRL/HER here.

Per step, for each primitive k we query its frozen actor at obs_k = (shared state)
+ (a synthesized goal_k), giving (mean_k, sigma_k). A learned gate selects a
primitive k* and emits a bounded residual; the action is

    a = tanh( mean_{k*} + sigma_{k*} (.) u ),   u = eps * raw / max(1, ||raw||)

i.e. the exact ball projection from train_residual_mab.barrier_residual, but with
the center/metric coming from the SELECTED primitive. Only the gate + residual are
learned; the primitives and their critics are frozen.

Ablation ladder (--barrier_type), the headline comparison:
  none       unconstrained RL: a = tanh(raw)  (no primitives; the baseline)
  single     one fixed primitive's ball (no selection); --single_primitive_idx
  euclidean  gated multi-primitive ball with Sigma = I (no per-dim sigma scaling)
  multi      gated multi-primitive diagonal-Mahalanobis ball (the method)

The per-primitive goal synthesis (`synth_goals`) is the planner interface (the role
a VLM would play in the full system); for the hallway it is hand-specified below.

Local smoke test (no checkpoints needed): pass --dry_run to build randomly
initialized dummy primitives so the whole pipeline compiles and runs:

  uv run train_constrained_composition.py --dry_run \
      --num_iterations 2 --grad_steps_per_iter 5 --num_envs 16 \
      --num_eval_envs 16 --episode_length 50

Real run (after the primitives are CRL-trained):

  uv run train_constrained_composition.py \
      --primitive_checkpoints PATH/navigate_final.pkl PATH/push_aside_final.pkl \
      --primitive_env_ids tidybot_navigate tidybot_push_aside \
      --env_id tidybot_hallway --barrier_type multi --exp_name comp_multi
"""

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
from typing import Tuple

from brax import envs
import flax.linen as nn
from flax.linen.initializers import variance_scaling
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

from train import Actor, residual_block, load_params, save_params, Args  # noqa: F401
from train_residual_mab import make_env
from envs.scripted_controllers import CONTROLLERS as SCRIPTED_CONTROLLERS

# Let the unpickler resolve `__main__.Args` for the primitives' sibling args.pkl.
import sys as _sys
try:
    setattr(_sys.modules["__main__"], "Args", Args)
except Exception:  # noqa: BLE001
    pass


@dataclass
class CompArgs:
    # --- primitives (parallel lists; checkpoints unused when --dry_run) ---
    primitive_checkpoints: Tuple[str, ...] = ()
    primitive_env_ids: Tuple[str, ...] = ("tidybot_navigate", "tidybot_push_aside")
    primitive_kind: Tuple[str, ...] = ("scripted", "scripted")
    """Per-slot ball source aligned to primitive_env_ids: 'scripted' (live
    controller center + Sigma), 'envelope' (fixed mu+Sigma from a success library;
    use a wider --epsilon ~2), or 'crl' (frozen actor mean + learned std)."""
    skill_ball_paths: Tuple[str, ...] = ()
    """Per-slot fitted-Sigma pkl from collect_scripted_skill_data.py (scripted
    slots). Empty/missing => fixed-Sigma fallback (also used for --dry_run)."""
    # --- composite task env ---
    env_id: str = "tidybot_hallway"
    state_dim: int = 20  # shared TidyBot state width (obs minus the goal tail)
    # --- barrier / composition ---
    barrier_type: str = "multi"  # none | single | euclidean | multi
    single_primitive_idx: int = 1  # used by barrier_type=single (default push_aside)
    epsilon: float = 0.3
    sigma_floor: float = 0.1
    # --- goal synthesis (the "planner interface" for the hallway) ---
    standoff: float = 0.4       # base stops this far (in y) short of the cube
    push_target_x: float = 0.45  # push the cube toward this x (near the wall)
    aside_thresh: float = 0.3    # |cube_x| beyond this counts as "lane cleared"
    push_rel_offset: float = 0.2  # base-relative aside target for the CRL push goal
    # --- gate/residual network ---
    net_width: int = 256
    net_depth: int = 4
    use_relu: int = 0
    # --- policy gradient ---
    lr: float = 3e-4
    explore_std: float = 0.3
    gate_entropy_coef: float = 0.01
    num_envs: int = 256
    num_iterations: int = 200
    grad_steps_per_iter: int = 50
    batch_size: int = 512
    # --- eval / io ---
    num_eval_envs: int = 128
    eval_every: int = 10
    success_threshold: float = 0.5
    episode_length: int = 500
    seed: int = 0
    exp_name: str = "comp"
    out_dir: str = ""
    dry_run: bool = False


class CompositionGate(nn.Module):
    """Shared trunk (train.Actor body style) -> (gate logits, residual raw).

    The residual head is zero-initialized so at init the action equals the selected
    primitive's mean (the residual starts at 0); the gate head is zero-initialized
    so selection starts uniform.
    """
    n_primitives: int
    action_size: int
    width: int = 256
    depth: int = 4
    use_relu: int = 0

    @nn.compact
    def __call__(self, x):
        lecun = variance_scaling(1 / 3, "fan_in", "uniform")
        zeros = nn.initializers.zeros
        normalize = lambda y: nn.LayerNorm()(y)
        activation = nn.relu if self.use_relu else nn.swish

        x = nn.Dense(self.width, kernel_init=lecun, bias_init=zeros)(x)
        x = normalize(x)
        x = activation(x)
        for _ in range(self.depth // 4):
            x = residual_block(x, self.width, normalize, activation)
        logits = nn.Dense(self.n_primitives, kernel_init=zeros, bias_init=zeros)(x)
        raw = nn.Dense(self.action_size, kernel_init=zeros, bias_init=zeros)(x)
        return logits, raw


def build_goal_fns(args):
    """Per-primitive goal synthesis from the composite (hallway) observation.

    Each returns a (B, goal_dim) goal that primitive's frozen actor is conditioned
    on. Hand-specified (the planner's job in the full system):
      navigate -> drive to a standoff just short of the cube until the lane is
                  cleared, then to the far-end goal.
      push_aside -> push the cube toward the +x wall to clear the lane.
    """
    sd = args.state_dim

    def navigate_goal(state, obs):
        cube_xy = state[:, 10:12]
        cube_x = state[:, 10]
        far_end = obs[:, sd:sd + 2]
        cleared = jnp.abs(cube_x) > args.aside_thresh
        standoff = cube_xy - jnp.array([0.0, args.standoff])
        return jnp.where(cleared[:, None], far_end, standoff)

    def push_goal(state, obs):
        cube_xy = state[:, 10:12]
        return cube_xy.at[:, 0].set(args.push_target_x)

    return {
        "tidybot_navigate": navigate_goal,
        "tidybot_push_aside": push_goal,
    }


def load_primitive_actor(checkpoint_path, action_size):
    """Build a frozen Actor + params from a CRL *_final.pkl (+ sibling _args.pkl)."""
    args_path = checkpoint_path.replace("_final.pkl", "_args.pkl")
    saved = load_params(args_path) if os.path.exists(args_path) else None

    def g(field, default):
        return getattr(saved, field, default) if saved is not None else default

    actor = Actor(
        action_size=action_size,
        network_width=g("actor_network_width", 256),
        network_depth=g("actor_depth", 4),
        skip_connections=g("actor_skip_connections", 0),
        use_relu=g("use_relu", 0),
    )
    _alpha, actor_params, _critic = load_params(checkpoint_path)
    return actor, actor_params


def main(args):
    print("JAX devices:", jax.devices(), flush=True)
    key = jax.random.PRNGKey(args.seed)

    # ---- composite env ----
    env = make_env(args.env_id)
    env = envs.training.wrap(env, episode_length=args.episode_length)
    env.step = jax.jit(env.step)
    reset = jax.jit(env.reset)
    obs_size = env.observation_size
    action_size = env.action_size
    n_prim = len(args.primitive_env_ids)
    goal_fns = build_goal_fns(args)
    print(f"[env] {args.env_id} obs={obs_size} act={action_size} "
          f"n_primitives={n_prim} barrier={args.barrier_type}", flush=True)

    sd = args.state_dim
    eps = args.epsilon
    btype = args.barrier_type
    uses_gate = btype in ("multi", "euclidean")
    sidx = args.single_primitive_idx

    # ---- primitive providers: each is stats(obs) -> (mean[B,A], sigma[B,A]) in
    # env-action space. 'scripted' = controller center + fitted/fixed Sigma;
    # 'crl' = frozen actor (tanh-mean) + learned std. ----
    def make_scripted(ctrl, goal_fn, sigma):
        def stats(obs):
            state = obs[:, :sd]
            mean = ctrl(state, goal_fn(state, obs))
            return mean, jnp.broadcast_to(sigma, mean.shape)
        return stats

    def make_crl(actor, params, goal_fn, base_relative):
        def stats(obs):
            state = obs[:, :sd]
            if base_relative:
                # Push primitive was trained base-at-origin / base-frozen. Rebuild its
                # obs in the base frame: base xy -> 0, cube/eef relative to the base,
                # goal = cube_rel + push_rel_offset. Hold the base in the output.
                B = state.shape[0]
                z = jnp.zeros((B, 1))
                off = jnp.concatenate([state[:, 0:1], state[:, 1:2], z], axis=-1)
                base_q = jnp.concatenate([z, z, state[:, 2:3]], axis=-1)
                cube = state[:, 10:13] - off
                eef = state[:, 13:16] - off
                st = jnp.concatenate(
                    [base_q, state[:, 3:10], cube, eef, state[:, 16:19], state[:, 19:20]],
                    axis=-1)
                goal = cube[:, 0:2] + jnp.array([args.push_rel_offset, 0.0])
                obs_k = jnp.concatenate([st, goal], axis=-1)
            else:
                obs_k = jnp.concatenate([state, goal_fn(state, obs)], axis=-1)
            m, ls = actor.apply(params, obs_k)
            mean = jnp.tanh(m)
            if base_relative:
                mean = mean.at[:, 0:3].set(0.0)  # hold base (trained base-frozen)
            return mean, jnp.maximum(jnp.exp(ls), args.sigma_floor)
        return stats

    def make_envelope(mu, sigma):
        # State-INDEPENDENT envelope from a success library: fixed center mu + spread
        # sigma. RL explores within mu +/- eps*sigma (use a wider --epsilon, ~2).
        def stats(obs):
            B = obs.shape[0]
            return (jnp.broadcast_to(mu, (B, mu.shape[0])),
                    jnp.broadcast_to(sigma, (B, sigma.shape[0])))
        return stats

    providers = []
    for i, eid in enumerate(args.primitive_env_ids):
        goal_fn = goal_fns[eid]
        kind = args.primitive_kind[i] if i < len(args.primitive_kind) else "scripted"
        if kind == "scripted":
            ctrl = SCRIPTED_CONTROLLERS[eid]
            if (not args.dry_run and i < len(args.skill_ball_paths)
                    and args.skill_ball_paths[i]):
                sigma = jnp.asarray(load_params(args.skill_ball_paths[i])["sigma"])
                src = args.skill_ball_paths[i]
            else:
                sigma = jnp.ones(action_size) * 0.3  # fixed fallback / dry-run
                src = "fixed(0.3)"
            providers.append(make_scripted(ctrl, goal_fn, sigma))
            print(f"[provider {i}] scripted {eid} sigma<-{src}", flush=True)
        elif kind == "envelope":
            if (args.dry_run or i >= len(args.skill_ball_paths)
                    or not args.skill_ball_paths[i]):
                mu = jnp.zeros(action_size)
                sigma = jnp.ones(action_size) * 0.5  # fixed fallback / dry-run
                src = "fixed(mu=0,sigma=0.5)"
            else:
                blob = load_params(args.skill_ball_paths[i])
                mu = jnp.asarray(blob["mu"])
                sigma = jnp.asarray(blob["sigma"])
                src = args.skill_ball_paths[i]
            providers.append(make_envelope(mu, sigma))
            print(f"[provider {i}] envelope {eid} mu/sigma<-{src}", flush=True)
        else:  # crl
            # Frozen-base push primitives (trained base-at-origin) need the
            # base-relative obs transform + base-hold to deploy in the mobile hallway.
            base_rel = eid in ("tidybot_push_aside", "tidybot_push_easy")
            if args.dry_run:
                actor = Actor(action_size=action_size, network_width=args.net_width,
                              network_depth=4, skip_connections=0, use_relu=args.use_relu)
                key, kp = jax.random.split(key)
                params = actor.init(kp, np.ones([1, obs_size]))
                src = "dummy"
            else:
                actor, params = load_primitive_actor(
                    args.primitive_checkpoints[i], action_size)
                src = args.primitive_checkpoints[i]
            providers.append(make_crl(actor, params, goal_fn, base_rel))
            print(f"[provider {i}] crl {eid} base_relative={base_rel} <-{src}", flush=True)

    def primitive_stats(obs):
        """(mean, sigma) per primitive at obs -> (B, N, A), env-action space."""
        means, sigmas = [], []
        for stats_fn in providers:
            m, s = stats_fn(obs)
            means.append(m)
            sigmas.append(s)
        return jnp.stack(means, axis=1), jnp.stack(sigmas, axis=1)

    def assemble(means, sigmas, onehot, raw_s):
        # Uniform action-space ball: a = clip(center + dx, -1, 1). center/metric come
        # from the gate-selected primitive (means are env-action space for both
        # provider kinds). 'none' = unconstrained policy directly in action space.
        if btype == "none":
            return jnp.clip(raw_s, -1.0, 1.0)
        if btype == "single":
            mean_sel, sigma_sel = means[:, sidx, :], sigmas[:, sidx, :]
        else:  # multi / euclidean
            mean_sel = jnp.einsum("bn,bna->ba", onehot, means)
            sigma_sel = jnp.einsum("bn,bna->ba", onehot, sigmas)
        norm = jnp.sqrt(jnp.sum(raw_s ** 2, axis=-1, keepdims=True) + 1e-8)
        u = eps * raw_s / jnp.maximum(1.0, norm)
        dx = u if btype == "euclidean" else sigma_sel * u
        return jnp.clip(mean_sel + dx, -1.0, 1.0)

    gate = CompositionGate(n_primitives=n_prim, action_size=action_size,
                           width=args.net_width, depth=args.net_depth,
                           use_relu=args.use_relu)
    key, gk = jax.random.split(key)
    gate_params = gate.init(gk, np.ones([1, obs_size]))
    comp_state = TrainState.create(
        apply_fn=gate.apply, params=gate_params,
        tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr)),
    )

    def policy_sample(params, obs, key):
        means, sigmas = primitive_stats(obs)
        logits, raw = gate.apply(params, obs)
        key, kk, kn = jax.random.split(key, 3)
        if uses_gate:
            sel = jax.random.categorical(kk, logits)
            onehot = jax.nn.one_hot(sel, n_prim)
        else:
            onehot = jax.nn.one_hot(jnp.full((obs.shape[0],), sidx), n_prim)
        raw_s = raw + args.explore_std * jax.random.normal(kn, raw.shape)
        action = assemble(means, sigmas, onehot, raw_s)
        return action, onehot, raw_s

    def policy_logprob(params, obs, onehot, raw_s):
        logits, raw = gate.apply(params, obs)
        logp = -0.5 * jnp.sum(((raw_s - raw) / args.explore_std) ** 2, axis=-1)
        ent = jnp.zeros(())
        if uses_gate:
            logp = logp + jnp.sum(jax.nn.log_softmax(logits) * onehot, axis=-1)
            p = jax.nn.softmax(logits)
            ent = -jnp.mean(jnp.sum(p * jax.nn.log_softmax(logits), axis=-1))
        return logp, ent

    @jax.jit
    def collect(params, env_state, key):
        def step(carry, _):
            env_state, key = carry
            key, ak = jax.random.split(key)
            action, onehot, raw_s = policy_sample(params, env_state.obs, ak)
            obs_t = env_state.obs
            env_state = env.step(env_state, action)
            return (env_state, key), (obs_t, onehot, raw_s, env_state.reward)

        (env_state, _), (obs, onehot, raw_s, rew) = jax.lax.scan(
            step, (env_state, key), None, length=args.episode_length)
        ep_return = jnp.sum(rew, axis=0)  # (N,) episodic (sparse) return
        return env_state, obs, onehot, raw_s, ep_return

    @jax.jit
    def update(comp_state, obs_b, onehot_b, raw_b, adv_b):
        def loss_fn(p):
            logp, ent = policy_logprob(p, obs_b, onehot_b, raw_b)
            pg = -jnp.mean(logp * adv_b)
            return pg - args.gate_entropy_coef * ent, (pg, ent)
        (loss, (pg, ent)), grad = jax.value_and_grad(loss_fn, has_aux=True)(comp_state.params)
        comp_state = comp_state.apply_gradients(grads=grad)
        return comp_state, loss, pg, ent

    @jax.jit
    def eval_rollout(params, env_state):
        def step(carry, _):
            env_state, sum_r, max_r, sel = carry
            means, sigmas = primitive_stats(env_state.obs)
            logits, raw = gate.apply(params, env_state.obs)
            if uses_gate:
                onehot = jax.nn.one_hot(jnp.argmax(logits, axis=-1), n_prim)
            else:
                onehot = jax.nn.one_hot(jnp.full((env_state.obs.shape[0],), sidx), n_prim)
            action = assemble(means, sigmas, onehot, raw)  # deterministic (no noise)
            env_state = env.step(env_state, action)
            r = env_state.reward
            return (env_state, sum_r + r, jnp.maximum(max_r, r), sel + onehot.mean(0)), None

        n = env_state.reward.shape[0]
        init = (env_state, jnp.zeros(n), jnp.zeros(n), jnp.zeros(n_prim))
        (_, sum_r, max_r, sel), _ = jax.lax.scan(
            step, init, None, length=args.episode_length)
        success = (max_r > args.success_threshold).astype(jnp.float32)
        return jnp.mean(sum_r), jnp.mean(success), sel / args.episode_length

    def run_eval(it):
        es = reset(jax.random.split(jax.random.PRNGKey(args.seed + 777), args.num_eval_envs))
        ret, succ, sel = eval_rollout(comp_state.params, es)
        sel_str = ",".join(f"{e}:{float(s):.2f}" for e, s in zip(args.primitive_env_ids, sel))
        print(f"[eval it={it:>3}] ret={float(ret):+.3f} success={float(succ):.3f} "
              f"selection[{sel_str}]", flush=True)
        return float(succ)

    # ---- training loop ----
    run_eval(0)
    for it in range(1, args.num_iterations + 1):
        key, ck = jax.random.split(key)
        env_state = reset(jax.random.split(ck, args.num_envs))  # fresh episodes
        env_state, obs, onehot, raw_s, ep_return = collect(comp_state.params, env_state, ck)
        adv = ep_return - jnp.mean(ep_return)
        T, N = obs.shape[0], obs.shape[1]
        obs_f = obs.reshape(T * N, -1)
        onehot_f = onehot.reshape(T * N, -1)
        raw_f = raw_s.reshape(T * N, -1)
        adv_f = jnp.broadcast_to(adv[None, :], (T, N)).reshape(T * N)

        last = (0.0, 0.0, 0.0)
        for _ in range(args.grad_steps_per_iter):
            key, ik = jax.random.split(key)
            idx = jax.random.randint(ik, (args.batch_size,), 0, T * N)
            comp_state, loss, pg, ent = update(
                comp_state, obs_f[idx], onehot_f[idx], raw_f[idx], adv_f[idx])
            last = (float(loss), float(pg), float(ent))
        print(f"[train it={it:>3}] ep_return_mean={float(jnp.mean(ep_return)):+.3f} "
              f"loss={last[0]:+.4f} pg={last[1]:+.4f} ent={last[2]:.3f}", flush=True)
        if args.eval_every and it % args.eval_every == 0:
            run_eval(it)

    run_eval(args.num_iterations)

    # ---- save ----
    out_dir = args.out_dir or "."
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(out_dir, f"composition_{args.env_id}_{args.exp_name}_{stamp}.pkl")
    save_params(out_path, {
        "gate_params": comp_state.params,
        "meta": {
            "env_id": args.env_id,
            "primitive_checkpoints": list(args.primitive_checkpoints),
            "primitive_env_ids": list(args.primitive_env_ids),
            "primitive_kind": list(args.primitive_kind),
            "skill_ball_paths": list(args.skill_ball_paths),
            "barrier_type": args.barrier_type,
            "epsilon": args.epsilon,
            "sigma_floor": args.sigma_floor,
            "net_width": args.net_width,
            "net_depth": args.net_depth,
        },
    })
    print(f"[save] {out_path}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(CompArgs))

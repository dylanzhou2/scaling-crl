import pickle
import jax
import jax.numpy as jnp
from flax import linen as nn
from brax.io import html

from envs.manipulation.arm_push_hard import ArmPushHard
from train import Actor, Args


# --------------------------------------------------
# Load checkpoint
# --------------------------------------------------

with open("scaling-crl_runs_arm_push_hard_1000_20260309-010323_final.pkl", "rb") as f:
    alpha_params, actor_params, critic_params = pickle.load(f)

with open("scaling-crl_runs_arm_push_hard_1000_20260309-010323_args.pkl", "rb") as f:
    args = pickle.load(f)

print("Checkpoint loaded")


# --------------------------------------------------
# Environment
# --------------------------------------------------

env = ArmPushHard(backend="mjx")

obs_size = env.observation_size
policy_action_size = 5
env_action_size = env.action_size

print("obs_size:", obs_size)
print("policy_action_size:", policy_action_size)
print("env_action_size:", env_action_size)


# --------------------------------------------------
# Actor
# --------------------------------------------------

actor = Actor(
    action_size=policy_action_size,
    network_width=args.actor_network_width,
    network_depth=args.actor_depth,
    skip_connections=args.actor_skip_connections,
    use_relu=args.use_relu,
)


# --------------------------------------------------
# Policy
# --------------------------------------------------

@jax.jit
def policy(params, obs):
    mean, _ = actor.apply(params, obs)
    return nn.tanh(mean)


# --------------------------------------------------
# JIT env functions
# --------------------------------------------------

env_step = jax.jit(env.step)
env_reset = jax.jit(env.reset)


# --------------------------------------------------
# Helper: freeze base in q/qd after each step
# --------------------------------------------------

def freeze_base(env_state, fixed_base_q):
    q = env_state.pipeline_state.q.at[:2].set(fixed_base_q)
    qd = env_state.pipeline_state.qd.at[:2].set(jnp.array([0.0, 0.0]))

    new_pipeline_state = env.pipeline_init(q, qd)
    new_obs = env._get_obs(
        new_pipeline_state,
        env_state.info["goal"],
        env_state.info["timestep"],
    )

    success, success_easy, success_hard = env._compute_goal_completion(
        new_obs, env_state.info["goal"]
    )
    env_state.metrics.update(
        success=success,
        success_easy=success_easy,
        success_hard=success_hard,
    )

    return env_state.replace(
        pipeline_state=new_pipeline_state,
        obs=new_obs,
    )


# --------------------------------------------------
# Drive base to target pose
# --------------------------------------------------

def drive_base_to_start(env_state, rollout_states, steps=20):
    target_base_action = jnp.array([0.0, 0.005])

    for _ in range(steps):
        arm_action = jnp.zeros(policy_action_size)
        action = jnp.concatenate([target_base_action, arm_action])

        env_state = env_step(env_state, action)
        rollout_states.append(env_state.pipeline_state)

    return env_state, target_base_action


# --------------------------------------------------
# Rollout
# --------------------------------------------------

def run_rollout(goal, seed):
    key = jax.random.PRNGKey(seed)

    env_state = env_reset(key)
    env_state = env.update_goal(env_state, goal)

    print("goal:", env_state.info["goal"])
    print("cube xyz:", env_state.pipeline_state.q[11:14])
    print("goal marker xyz:", env_state.pipeline_state.q[18:21])
    print("base q at reset:", env_state.pipeline_state.q[:2])

    rollout_states = []
    total_reward = 0.0

    # --------------------------------------------------
    # Phase 1: drive base
    # --------------------------------------------------

    env_state, fixed_base_action = drive_base_to_start(env_state, rollout_states)
    fixed_base_q = env_state.pipeline_state.q[:2]

    print("base q after drive:", fixed_base_q)

    # pause while holding the base fixed
    for _ in range(20):
        arm_action = jnp.zeros(policy_action_size)
        action = jnp.concatenate([fixed_base_action, arm_action])
        env_state = env_step(env_state, action)
        env_state = freeze_base(env_state, fixed_base_q)
        rollout_states.append(env_state.pipeline_state)

    # --------------------------------------------------
    # Phase 2: push policy while base is clamped
    # --------------------------------------------------

    for t in range(env.episode_length):
        obs = jnp.expand_dims(env_state.obs, axis=0)

        arm_action = policy(actor_params, obs)[0]
        action = jnp.concatenate([fixed_base_action, arm_action])

        env_state = env_step(env_state, action)
        env_state = freeze_base(env_state, fixed_base_q)

        rollout_states.append(env_state.pipeline_state)

        total_reward += env_state.reward.item()

        if env_state.metrics["success_hard"]:
            print(f"Goal reached early at step {t}")
            break

    print("Held base target:", fixed_base_action)
    print("Held base q:", fixed_base_q)
    print("Goal:", goal, "Episode reward:", total_reward)
    return rollout_states


# --------------------------------------------------
# Run episodes
# --------------------------------------------------

goals = [
    jnp.array([0.30, 0.70, 0.03]),
]

all_rollout_states = []

for i, goal in enumerate(goals):
    episode_states = run_rollout(goal, seed=i)
    all_rollout_states.extend(episode_states)

    for _ in range(40):
        all_rollout_states.append(episode_states[-1])


# --------------------------------------------------
# Render HTML
# --------------------------------------------------

html_string = html.render(env.sys, all_rollout_states)

with open("arm_push_hard_mobile_rollout3.html", "w") as f:
    f.write(html_string)

print("Saved rollout to arm_push_hard_mobile_rollout2.html")
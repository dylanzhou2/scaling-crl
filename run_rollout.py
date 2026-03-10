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
print(env._get_xml_path())

obs_size = env.observation_size
action_size = env.action_size

print("obs_size:", obs_size)
print("action_size:", action_size)


# --------------------------------------------------
# Actor
# --------------------------------------------------

actor = Actor(
    action_size=action_size,
    network_width=args.actor_network_width,
    network_depth=args.actor_depth,
    skip_connections=args.actor_skip_connections,
    use_relu=args.use_relu,
)


# --------------------------------------------------
# JIT policy
# --------------------------------------------------

@jax.jit
def policy(params, obs):
    mean, _ = actor.apply(params, obs)
    return nn.tanh(mean)


# --------------------------------------------------
# JIT env step
# --------------------------------------------------

env_step = jax.jit(env.step)
env_reset = jax.jit(env.reset)


# --------------------------------------------------
# Rollout
# --------------------------------------------------

key = jax.random.PRNGKey(0)
env_state = env_reset(key)

rollout_states = []
total_reward = 0

for t in range(1000):

    obs = jnp.expand_dims(env_state.obs, axis=0)

    action = policy(actor_params, obs)[0]
    # print("action:", action)

    env_state = env_step(env_state, action)

    rollout_states.append(env_state.pipeline_state)

    total_reward += env_state.reward.item()

    if env_state.done:
        break


print("Episode reward:", total_reward)


# --------------------------------------------------
# Render
# --------------------------------------------------

html_string = html.render(env.sys, rollout_states)

with open("arm_push_rollout.html", "w") as f:
    f.write(html_string)

print("Saved rollout to arm_push_rollout.html")
"""
Oracle controller sanity check for TidyBotPushEasy.

Answers: can any non-random policy achieve reward > 0 without training?
If not, the controller/geometry is misconfigured regardless of training.

Checks:
  1. EEF and cube positions at reset (are they even in the same ballpark?)
  2. A scripted "move base toward cube, then hold" controller for 500 steps
  3. Whether any reward is achieved and what the min EEF-cube distance was

Usage: uv run oracle_controller_test.py
"""

import jax
import jax.numpy as jnp
import numpy as np

from envs.mobile_manipulation.tidybot_push_easy import TidyBotPushEasy

print("Initializing TidyBotPushEasy...")
env = TidyBotPushEasy(backend="mjx")
reset = jax.jit(env.reset)
step = jax.jit(env.step)

NUM_EPISODES = 5
EPISODE_LENGTH = 500

rng = jax.random.PRNGKey(0)

print(f"\n{'='*60}")
print("Geometry check at reset (episode 0)")
print(f"{'='*60}")

rng, ep_key = jax.random.split(rng)
state = reset(ep_key)
obs = np.array(state.obs)

# obs layout: base(0-2), arm(3-9), cube_pos(10-12), eef_pos(13-15), eef_vel(16-18), finger(19), goal(20-22)
eef_pos = obs[13:16]
cube_pos = obs[10:13]
goal_pos = obs[20:23]
base_pos = obs[0:3]

print(f"Base position (x,y,th):  {base_pos}")
print(f"EEF position  (x,y,z):   {eef_pos}")
print(f"Cube position (x,y,z):   {cube_pos}")
print(f"Goal position (x,y,z):   {goal_pos}")
print(f"EEF-to-cube distance:    {np.linalg.norm(eef_pos - cube_pos):.4f} m")
print(f"Cube-to-goal distance:   {np.linalg.norm(cube_pos - goal_pos):.4f} m")

if eef_pos[2] > 0.5:
    print("\n[WARNING] EEF z > 0.5m — arm appears to be pointing up, NOT near cube.")
    print("  This suggests the arm default configuration needs to be set explicitly")
    print("  (like ArmPushEasy's arm_q_default) for the robot to reach the cube.")
elif np.linalg.norm(eef_pos[:2] - cube_pos[:2]) > 0.5:
    print("\n[WARNING] EEF is far from cube in XY — base may need to navigate first.")
else:
    print("\n[OK] EEF is in reasonable proximity to cube.")

print(f"\n{'='*60}")
print(f"Oracle controller rollouts ({NUM_EPISODES} episodes)")
print(f"Strategy: move base toward cube for 200 steps, then hold")
print(f"{'='*60}")

all_max_rewards = []
all_min_eef_cube_dists = []

for ep in range(NUM_EPISODES):
    rng, ep_key = jax.random.split(rng)
    state = reset(ep_key)

    max_reward = 0.0
    min_eef_cube_dist = float("inf")

    for t in range(EPISODE_LENGTH):
        obs = np.array(state.obs)
        base_pos = obs[0:3]
        cube_pos = obs[10:13]
        eef_pos = obs[13:16]
        goal_pos = obs[20:23]

        eef_cube_dist = float(np.linalg.norm(eef_pos - cube_pos))
        min_eef_cube_dist = min(min_eef_cube_dist, eef_cube_dist)

        # Oracle strategy: move base so robot is behind the cube relative to goal
        cube_to_goal = goal_pos[:2] - cube_pos[:2]
        push_dir = cube_to_goal / (np.linalg.norm(cube_to_goal) + 1e-6)
        # Position robot behind the cube (opposite to goal direction)
        target_base_xy = cube_pos[:2] - push_dir * 0.3

        base_error = target_base_xy - base_pos[:2]
        base_action = np.clip(base_error / 0.2, -1, 1)  # proportional, matches 0.2 scale

        # Arm: try to push down and forward — use a fixed "pushing" configuration
        # Just hold arm at zero for now to see if base navigation alone can help
        arm_action = np.zeros(7)

        # Gripper: open
        gripper_action = np.array([-1.0])

        action = np.concatenate([base_action, [0.0], arm_action, gripper_action])
        action_jnp = jnp.array(action)

        rng, step_key = jax.random.split(rng)
        state = step(state, action_jnp)

        reward = float(state.reward)
        max_reward = max(max_reward, reward)

        if reward > 0:
            print(f"  ep={ep} t={t:3d}  REWARD={reward:.2f}  eef-cube={eef_cube_dist:.3f}m  cube-goal={np.linalg.norm(cube_pos - goal_pos):.3f}m")

    all_max_rewards.append(max_reward)
    all_min_eef_cube_dists.append(min_eef_cube_dist)
    print(f"  ep={ep}  max_reward={max_reward:.2f}  min_eef-cube_dist={min_eef_cube_dist:.4f}m")

print(f"\n{'='*60}")
print("Summary")
print(f"{'='*60}")
print(f"Episodes with any reward > 0:  {sum(r > 0 for r in all_max_rewards)}/{NUM_EPISODES}")
print(f"Min EEF-cube dist (best ep):   {min(all_min_eef_cube_dists):.4f} m")
print(f"Mean min EEF-cube dist:        {np.mean(all_min_eef_cube_dists):.4f} m")

if min(all_min_eef_cube_dists) > 0.3:
    print("\n[DIAGNOSIS] EEF never gets within 0.3m of cube.")
    print("  Root cause is likely the arm default configuration (zero joints = arm pointing up).")
    print("  Fix: set arm_q_default in TidyBotPushEasy._get_initial_state to a")
    print("  pre-grasp pose that positions the EEF near table height.")
elif sum(r > 0 for r in all_max_rewards) == 0:
    print("\n[DIAGNOSIS] EEF gets near cube but no reward. Check success threshold or goal geometry.")
else:
    print("\n[OK] Oracle controller achieves reward — controller is wired correctly.")
    print("  The issue is purely the learned policy (training config), not the controller setup.")

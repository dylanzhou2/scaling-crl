import numpy as np
import time
import pickle
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
from mujoco import mjx

import sys
import os

# Add the parent directory (scaling-crl) to Python's module search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your custom modules
from envs.manipulation.arm_push_hard import ArmPushHard
from train import Actor, Args

def main():
    # --------------------------------------------------
    # 1. Load checkpoint
    # --------------------------------------------------
    with open("/Users/dylanzhou/Downloads/scaling-crl_runs_arm_push_hard_1000_20260309-010323_final.pkl", "rb") as f:
        alpha_params, actor_params, critic_params = pickle.load(f)

    with open("/Users/dylanzhou/Downloads/scaling-crl_runs_arm_push_hard_1000_20260309-010323_args.pkl", "rb") as f:
        args = pickle.load(f)

    print("Checkpoint loaded")

    # --------------------------------------------------
    # 2. Setup Environment and Actor
    # --------------------------------------------------
    env = ArmPushHard(backend="mjx")
    
    actor = Actor(
        action_size=env.action_size,
        network_width=args.actor_network_width,
        network_depth=args.actor_depth,
        skip_connections=args.actor_skip_connections,
        use_relu=args.use_relu,
    )

    @jax.jit
    def policy(params, obs):
        mean, _ = actor.apply(params, obs)
        return jax.nn.tanh(mean)

    env_step = jax.jit(env.step)
    env_reset = jax.jit(env.reset)

    def sync_viewer(mj_model, mj_data, pipeline_state):
        # Copy JAX arrays to CPU numpy arrays
        mj_data.qpos[:] = np.array(pipeline_state.q)
        mj_data.qvel[:] = np.array(pipeline_state.qd)
        # Forward kinematics so the viewer updates object positions
        mujoco.mj_forward(mj_model, mj_data)
    
    # We create a JITted helper to update the goal AND the observation
    @jax.jit
    def apply_user_goal(state, new_goal):
        # env.update_goal handles state.info['goal'] and pipeline_state visualization
        state = env.update_goal(state, new_goal)
        # We must manually refresh the observation so the policy sees the new goal
        new_obs = env._get_obs(state.pipeline_state, new_goal, state.info["timestep"])
        return state.replace(obs=new_obs)

    # --------------------------------------------------
    # 3. Initialize State & MuJoCo Sync
    # --------------------------------------------------
    key = jax.random.PRNGKey(0)
    env_state = env_reset(key)

    mj_model = env.sys.mj_model 
    mj_data = mujoco.MjData(mj_model)

    # Sync initial JAX state to the CPU viewer
    # mujoco.mjx.get_data(mj_model, mj_data, env_state.pipeline_state.mjx_data)
    sync_viewer(mj_model, mj_data, env_state.pipeline_state)

    # --------------------------------------------------
    # 4. Interactive Viewer Logic
    # --------------------------------------------------
    execution_phase = False
    
    # Extract the initial starting goal from the JAX state
    current_goal = np.array(env_state.info["goal"])

    def key_callback(keycode):
        nonlocal execution_phase, current_goal, env_state
        
        # 1. Print every key press so we know macOS isn't blocking the inputs!
        print(f"Key registered: {keycode}")
        
        step_size = 0.05  # Increased from 0.02 to 0.05 for much more visible jumps
        
        if keycode == 32:  # Spacebar
            execution_phase = True
            print("Spacebar pressed. Locking goal and starting execution!")
            
        elif keycode in [265, 87]:  # Up Arrow or 'W'
            current_goal[0] += step_size
        elif keycode in [264, 83]:  # Down Arrow or 'S'
            current_goal[0] -= step_size
        elif keycode in [263, 65]:  # Left Arrow or 'A'
            current_goal[1] += step_size
        elif keycode in [262, 68]:  # Right Arrow or 'D'
            current_goal[1] -= step_size

        if keycode in [265, 264, 263, 262, 87, 83, 65, 68]:
            print(f"Moving goal to new coordinates: {current_goal}")
            # Instantly update JAX and copy back to CPU
            env_state = apply_user_goal(env_state, jnp.array(current_goal))
            sync_viewer(mj_model, mj_data, env_state.pipeline_state)
            
            # 2. Force MuJoCo to redraw the screen the exact millisecond the key is pressed
            viewer.sync()

    print("\n" + "="*50)
    print("PHASE 1: Use WASD or ARROW KEYS to move the goal.")
    print("Press SPACEBAR to confirm and start the policy.")
    print("="*50 + "\n")

    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
        sync_viewer(mj_model, mj_data, env_state.pipeline_state)
        viewer.sync()

        # --- PHASE 1: PAUSE & MOVE WITH KEYS ---
        while viewer.is_running() and not execution_phase:
            viewer.sync()
            time.sleep(0.05)
            
        if not viewer.is_running():
            return

        print(f"Goal locked at coordinates: {current_goal}")

        # --- PHASE 2: EXECUTION ---
        print("\nPHASE 2: Executing push policy...")
        step_count = 0
        max_steps = 1000

        while viewer.is_running() and step_count < max_steps and not env_state.done:
            step_start = time.time()

            obs = jnp.expand_dims(env_state.obs, axis=0)
            action = policy(actor_params, obs)[0]
            env_state = env_step(env_state, action)

            # Sync JAX state back to CPU viewer for rendering
            # mujoco.mjx.get_data(mj_model, mj_data, env_state.pipeline_state.mjx_data)
            sync_viewer(mj_model, mj_data, env_state.pipeline_state)
            viewer.sync()

            step_count += 1

            # Enforce real-time playback speed (Brax dt is usually n_frames * opt.timestep)
            # In your config: opt.timestep is 0.002 and n_frames is 25, so dt = 0.05 seconds
            dt = env.dt 
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(float(time_until_next_step))

        print("\nExecution finished.")
        print("PHASE 3: Window will remain open indefinitely. Close window to exit.")

        # --- PHASE 3: KEEP WINDOW OPEN ---
        while viewer.is_running():
            time.sleep(0.1)

if __name__ == "__main__":
    main()
from brax import base
from envs.manipulation.tidybot_envs import TidyBotEnv
import jax
from jax import numpy as jnp

class TidyBotPushEasy(TidyBotEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Verification prints confirm the layout during initialization
        print(f"\n--- {self.env_name} Layout Verification ---")
        print("Link names and indices:")
        for i, name in enumerate(self.sys.link_names):
            print(f"  Index {i}: {name}")

        print("Position vector size (q):", self.sys.q_size())
        print("Velocity vector size (qd):", self.sys.qd_size())

        try:
            cube_idx = self.sys.link_names.index("cube")
            goal_idx = self.sys.link_names.index("goal_marker")
            print(f"Confirmed: Cube Link Index = {cube_idx}, Goal Link Index = {goal_idx}")
        except ValueError:
            print("Warning: 'cube' or 'goal_marker' links not found in sys.link_names.")

    def _get_xml_path(self):
        return "envs/assets/tidybot_push_easy.xml"

    def _set_environment_attributes(self):
        self.env_name = "tidybot_push_easy"
        self.episode_length = 50

        # Based on q=32 verification:
        # Robot (0-17): Base(3) + Arm(7) + Gripper(8)
        # Cube (18-24): 7-dim freejoint starts at index 18
        # Goal (25-31): 7-dim freejoint starts at index 25
        self.cube_q_idx = 18 
        self.goal_q_idx = 25 

        # Observation mapping within the 20-dim state vector (before goal is appended)
        # indices: base(0-2), arm(3-9), cube_pos(10-12), eef(13-18), finger(19)
        self.goal_indices = jnp.array([10, 11, 12]) 
        self.completion_goal_indices = self.goal_indices
        self.state_dim = 20

        self.arm_noise_scale = 0.05
        self.cube_noise_scale = 0.1
        self.goal_noise_scale = 0.1

    def _get_initial_state(self, rng):
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        q = self.sys.init_q
        
        # 1. Randomize Cube Position (xy) at index 18
        cube_q_xy = q[self.cube_q_idx : self.cube_q_idx+2] + \
                    self.cube_noise_scale * jax.random.uniform(subkey1, [2], minval=-1)
        q = q.at[self.cube_q_idx : self.cube_q_idx+2].set(cube_q_xy)

        # 2. Add noise to Arm joints (indices 3-9)
        arm_q = q[3:10] + self.arm_noise_scale * jax.random.uniform(subkey2, [7], minval=-1)
        q = q.at[3:10].set(arm_q)

        qd = jnp.zeros([self.sys.qd_size()])
        return q, qd

    def _get_initial_goal(self, pipeline_state: base.State, rng):
        rng, subkey = jax.random.split(rng)
        # Define a target region in front of the robot
        cube_goal_pos = jnp.array([0.6, 0.1, 0.03]) + jnp.array(
            [self.goal_noise_scale, self.goal_noise_scale, 0]
        ) * jax.random.uniform(subkey, [3], minval=-1)
        return cube_goal_pos

    def _compute_goal_completion(self, obs, goal):
        current_cube_pos = obs[self.completion_goal_indices]
        dist = jnp.linalg.norm(current_cube_pos - goal[:3])

        success = jnp.array(dist < 0.1, dtype=float)
        success_easy = jnp.array(dist < 0.3, dtype=float)
        success_hard = jnp.array(dist < 0.03, dtype=float)
        return success, success_easy, success_hard

    def _update_goal_visualization(self, pipeline_state: base.State, goal: jax.Array) -> base.State:
        # Move the visual ghost object (index 25) to the goal position
        updated_q = pipeline_state.q.at[self.goal_q_idx : self.goal_q_idx+3].set(goal[:3])
        return pipeline_state.replace(qpos=updated_q)

    def _get_obs(self, pipeline_state: base.State, goal: jax.Array, timestep) -> jax.Array:
        """
        Observation space (23-dim):
        - base_q (3-dim): x, y, theta
        - arm_q (7-dim): joint angles
        - cube_pos (3-dim): current cube location (from index 18)
        - eef (6-dim): position and velocity of bracelet_link
        - finger (1-dim): distance between right_follower (11) and left_follower (15)
        - goal (3-dim): target location
        """
        base_q = pipeline_state.q[:3]
        arm_q = pipeline_state.q[3:10]
        cube_pos = pipeline_state.q[self.cube_q_idx : self.cube_q_idx+3]
        
        eef_pos = pipeline_state.x.pos[self.eef_index]
        eef_vel = pipeline_state.xd.vel[self.eef_index]
        
        # Fixed: Using indices 11 and 15 from the verification output
        finger_dist = jnp.linalg.norm(
            pipeline_state.x.pos[11] - pipeline_state.x.pos[15], keepdims=True
        )

        return jnp.concatenate([base_q, arm_q, cube_pos, eef_pos, eef_vel, finger_dist, goal])
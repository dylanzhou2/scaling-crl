from brax import base
from envs.manipulation.tidybot_envs import TidyBotEnv
import jax
from jax import numpy as jnp

class TidyBotPushHard(TidyBotEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 1. Dynamically find link indices to prevent index-shift errors
        try:
            self.cube_link_idx = self.sys.link_names.index("cube")
            self.goal_link_idx = self.sys.link_names.index("goal_marker")
            self.eef_index = self.sys.link_names.index("bracelet_link")
            
            # Fingertip indices for finger_dist calculation
            self.left_finger_idx = self.sys.link_names.index("left_follower")
            self.right_finger_idx = self.sys.link_names.index("right_follower")
            
        except ValueError as e:
            print(f"Index Error: {e}. Check link names in tidybot_push_hard.xml")

        # 2. Keep a single verification summary for the master process
        print(f"--- {self.env_name} Initialized ---")
        print(f"Indices: Cube={self.cube_link_idx}, Goal={self.goal_link_idx}, EEF={self.eef_index}")

    def _get_xml_path(self):
        # Updated to point to the hard mode XML
        return "envs/assets/tidybot_push_hard.xml"

    def _set_environment_attributes(self):
        self.env_name = "tidybot_push_hard"
        self.episode_length = 80  # Increased for hard task

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

        # Hard mode scales: larger regions for cube and goal, zero arm noise
        self.arm_noise_scale = 0.0
        self.cube_noise_scale = 0.25
        self.goal_noise_scale = 0.25

    def _get_initial_state(self, rng):
        rng, subkey1, subkey2 = jax.random.split(rng, 3)
        q = self.sys.init_q
        
        # 1. Randomize Cube Position (xy) with larger hard-mode bounds
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
        # Randomize over a larger target region in front of the robot
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
        # Fixed: Use .q instead of .qpos for Brax/MJX State objects
        updated_q = pipeline_state.q.at[self.goal_q_idx : self.goal_q_idx+3].set(goal[:3])
        return pipeline_state.replace(q=updated_q)

    def _get_obs(self, pipeline_state: base.State, goal: jax.Array, timestep) -> jax.Array:
        # generalized coordinates
        base_q = pipeline_state.q[:3]
        arm_q = pipeline_state.q[3:10]
        cube_pos = pipeline_state.q[self.cube_q_idx : self.cube_q_idx+3]
        
        # spatial coordinates (x.pos)
        eef_pos = pipeline_state.x.pos[self.eef_index]
        eef_vel = pipeline_state.xd.vel[self.eef_index]
        
        # Normalized finger distance
        finger_dist = jnp.linalg.norm(
            pipeline_state.x.pos[self.left_finger_idx] - pipeline_state.x.pos[self.right_finger_idx], 
            keepdims=True
        ) + 1e-8

        # Total: 3 + 7 + 3 + 3 + 3 + 1 = 20 (state) + 3 (goal) = 23 total
        return jnp.concatenate([base_q, arm_q, cube_pos, eef_pos, eef_vel, finger_dist, goal])
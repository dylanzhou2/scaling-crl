from brax import base
from envs.manipulation.arm_envs import ArmEnvs
import jax
from jax import numpy as jnp

class TidyBotEnv(ArmEnvs):
    """
    Base class for Tidybot (Mobile Base + Kinova Gen3 + Robotiq 2f-85).
    """

    def __init__(self, backend="mjx", **kwargs):
        super().__init__(backend=backend, **kwargs)
        # Fix 1: Use link_names to find index instead of body_name2id
        self.eef_index = self.sys.link_names.index("bracelet_link")
        # Fix 2: Correct attribute name is jnt_range
        self.arm_joint_range = self.sys.jnt_range[3:10] 

    def _get_arm_angles(self, pipeline_state: base.State) -> jax.Array:
        # Return 10 joints (0-2 base, 3-9 arm) for unified control context
        return pipeline_state.q[:10]

    def _convert_action_to_actuator_input_joint_angle(
        self, action: jax.Array, arm_angles: jax.Array, delta_control: bool = True
    ) -> jax.Array:
        action = jnp.clip(action, -1.0, 1.0)

        # 1. Base Actuators (0, 1, 2)
        base_current = arm_angles[:3] 
        base_action = base_current + action[:3] * 0.2

        # 2. Arm Actuators (3-9)
        arm_action_raw = action[3:10]
        min_val, max_val = self.arm_joint_range[:, 0], self.arm_joint_range[:, 1]
        offset, multiplier = (min_val + max_val) / 2, (max_val - min_val) / 2

        if delta_control:
            normalized = (arm_angles[3:10] - offset) / multiplier
            arm_action_raw = jnp.clip(normalized + arm_action_raw * 0.25, -1.0, 1.0)

        arm_action = offset + arm_action_raw * multiplier

        # 3. Gripper Actuator (10)
        gripper_action = jnp.where(action[10] > 0, 0.0, 255.0)[None]

        return jnp.concatenate([base_action, arm_action, gripper_action])

    def _get_obs(self, pipeline_state: base.State, goal: jax.Array, timestep) -> jax.Array:
        base_q = pipeline_state.q[:3]
        arm_q = pipeline_state.q[3:10]
        
        # This will be overridden or called by subclasses with correct cube indices
        cube_pos = pipeline_state.q[18:21] 
        
        eef_pos = pipeline_state.x.pos[self.eef_index]
        eef_vel = pipeline_state.xd.vel[self.eef_index]
        
        # Fix 3: Updated finger indices from verification (11: right_follower, 15: left_follower)
        finger_dist = jnp.linalg.norm(
            pipeline_state.x.pos[11] - pipeline_state.x.pos[15], keepdims=True
        )

        return jnp.concatenate([base_q, arm_q, cube_pos, eef_pos, eef_vel, finger_dist, goal])
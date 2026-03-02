from brax import base
from envs.manipulation.arm_envs import ArmEnvs
import jax
from jax import numpy as jnp

class TidyBotEnv(ArmEnvs):
    """
    Base class for Tidybot (Mobile Base + Kinova Gen3 + Robotiq 2f-85).
    
    Architecture:
    - Actuators (11 total): [base_x, base_y, base_th, arm_1...7, fingers]
    - qpos Layout: [base_x, base_y, base_th, arm_1...7, gripper_linkage...]
    """

    def __init__(self, backend="mjx", **kwargs):
        super().__init__(backend=backend, **kwargs)
        # End-effector (bracelet_link) is the last link before the gripper
        self.eef_index = self.sys.body_name2id("bracelet_link")
        # Cache arm ranges
        self.arm_joint_range = self.sys.joint_range[3:10] 

    def _get_arm_angles(self, pipeline_state: base.State) -> jax.Array:
        # Skips the 3 mobile base joints
        return pipeline_state.q[3:10]

    def _convert_action_to_actuator_input_joint_angle(
        self, action: jax.Array, arm_angles: jax.Array, delta_control: bool = True
    ) -> jax.Array:
        """
        Unified Whole-Body Control:
        Action Space (11-dim): [base_x, base_y, base_th, arm_1..7, gripper]
        """
        action = jnp.clip(action, -1.0, 1.0)

        # 1. Base Actuators (0, 1, 2)
        # Assuming the policy targets a delta in the base's position
        base_current = arm_angles[:3] # Using indices 0-2 from full qpos
        base_action = base_current + action[:3] * 0.2

        # 2. Arm Actuators (3-9)
        arm_action_raw = action[3:10]
        min_val, max_val = self.arm_joint_range[:, 0], self.arm_joint_range[:, 1]
        offset, multiplier = (min_val + max_val) / 2, (max_val - min_val) / 2

        if delta_control:
            # arm_angles[3:] captures the Kinova joints
            normalized = (arm_angles[3:10] - offset) / multiplier
            arm_action_raw = jnp.clip(normalized + arm_action_raw * 0.25, -1.0, 1.0)

        arm_action = offset + arm_action_raw * multiplier

        # 3. Gripper Actuator (10) - Single scalar for 2f-85
        gripper_action = jnp.where(action[10] > 0, 0.0, 255.0)[None]

        return jnp.concatenate([base_action, arm_action, gripper_action])

    def _get_obs(self, pipeline_state: base.State, goal: jax.Array, timestep) -> jax.Array:
        """Standard Tidybot Observation Layout."""
        base_q = pipeline_state.q[:3]
        arm_q = pipeline_state.q[3:10]
        eef_pos = pipeline_state.x.pos[self.eef_index]
        eef_vel = pipeline_state.xd.vel[self.eef_index]
        
        # Scalar finger distance based on link indices 12 and 18 from tidybot.xml
        finger_dist = jnp.linalg.norm(
            pipeline_state.x.pos[12] - pipeline_state.x.pos[18], keepdims=True
        )

        return jnp.concatenate([base_q, arm_q, eef_pos, eef_vel, finger_dist, goal])
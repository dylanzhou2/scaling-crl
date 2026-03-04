import numpy as np
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jnp
from envs.mobile_manipulation.maze_utils import generate_procedural_grid, make_tidybot_maze

class TidyBotMaze(PipelineEnv):
    def __init__(self, width=11, height=11, include_clutter=True, backend="mjx", **kwargs):
        # 1. Setup Procedural Environment
        self.grid = generate_procedural_grid(width, height)
        self.grid_jax = jnp.array(self.grid)
        self.grid_obs_static = jnp.array(self.grid).flatten()
        xml_string, self.goal_pos = make_tidybot_maze(
            self.grid, size_scaling=4.0, include_clutter=include_clutter
        )
        sys = mjcf.loads(xml_string)
        
        # 2. Define Tucked Arm Configuration
        # [joint_1...7] + fingers
        self.tucked_arm_q = jnp.array([0, -1.0, 0, -2.0, 0, 1.0, 0, 0])
        
        super().__init__(sys=sys, backend=backend, n_frames=5, **kwargs)

    def reset(self, rng: jax.Array) -> State:
        # Use jnp.nonzero with a fixed size to satisfy JIT compilation
        # This prevents the ConcretizationTypeError by fixing the output shape
        rows, cols = jnp.nonzero(self.grid_jax == 2, size=1)
        
        # Extract the first coordinate found
        start_indices = jnp.array([rows[0], cols[0]])
        start_pos = start_indices * 4.0
        
        # Initialize the robot base position at these coordinates
        q = jnp.zeros(self.sys.q_size()).at[:2].set(start_pos)
        qd = jnp.zeros(self.sys.qd_size())
        
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        return State(pipeline_state, obs, jnp.zeros(()), jnp.zeros(()), {}, {"step": 0})

    def step(self, state: State, action: jax.Array) -> State:
        # Action is 3-dim: [base_x, base_y, base_th]
        # Map to 11 actuators: [3 base actuators] + [7 arm actuators] + [1 gripper]
        base_action = jnp.clip(action, -1.0, 1.0) * 0.2
        full_action = jnp.concatenate([base_action, self.tucked_arm_q])
        
        pipeline_state = self.pipeline_step(state.pipeline_state, full_action)
        obs = self._get_obs(pipeline_state)
        
        # Reward: Navigation Success
        dist = jnp.linalg.norm(pipeline_state.x.pos[0, :2] - self.goal_pos[:2])
        reward = -dist - 0.1 * jnp.sum(jnp.square(action))  # Dist + Control Cost
        
        done = jnp.array(dist < 0.5, dtype=float)
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        # Follows AntMaze pattern: Robot State + Velocities + Global Map
        base_pos = pipeline_state.q[:3]
        base_vel = pipeline_state.qd[:3]
        
        # Flattened Global Map (Occupancy Grid)
        grid_obs = self.grid_obs_static
        
        return jnp.concatenate([
            base_pos,           # 3: x, y, theta
            base_vel,           # 3: vx, vy, vtheta
            self.goal_pos[:2],  # 2: target x, y
            grid_obs            # Global knowledge
        ])
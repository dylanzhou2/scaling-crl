import numpy as np
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jnp
from envs.mobile_manipulation.maze_utils import generate_procedural_grid, make_tidybot_mocap_maze

class TidyBotMaze(PipelineEnv):
    def __init__(self, width=11, height=11, include_clutter=False, backend="mjx", num_mazes=1000, **kwargs):
        self.width = width
        self.height = height
        self.size_scaling = 4.0

        # 1. Pre-generate a Bank of Mazes in Python
        mazes = []
        for _ in range(num_mazes):
            mazes.append(generate_procedural_grid(width, height))
        self.maze_bank = jnp.array(mazes)

        # 2. Create Maximal Environment (All walls exist as MOCAP bodies)
        xml_string = make_tidybot_mocap_maze(width, height, size_scaling=self.size_scaling)
        sys = mjcf.loads(xml_string)

        self.nu = sys.nu
        assert self.nu == 11, f"Unexpected actuator count: {self.nu}"

        self.arm_rest = jnp.array([0, -1.0, 0, -2.0, 0, 1.0, 0])
        self.gripper_rest = jnp.array([0.0])        
        super().__init__(sys=sys, backend=backend, n_frames=5, **kwargs)

    @property
    def action_size(self):
        """Forces the RL policy to output only 3 values (base_x, base_y, base_th)."""
        return 3

    def reset(self, rng: jax.Array) -> State:
        rng, maze_rng = jax.random.split(rng)

        # Sample a random maze from the bank
        maze_idx = jax.random.randint(maze_rng, (), 0, self.maze_bank.shape[0])
        grid = self.maze_bank[maze_idx]

        # Robot Start
        rows, cols = jnp.nonzero(grid == 2, size=1)
        start_pos = jnp.array([rows[0], cols[0]]) * self.size_scaling

        # Goal Position
        g_rows, g_cols = jnp.nonzero(grid == 3, size=1)
        goal_pos = jnp.array([g_rows[0], g_cols[0]]) * self.size_scaling

        q = jnp.zeros(self.sys.q_size()).at[:2].set(start_pos)
        qd = jnp.zeros(self.sys.qd_size())

        pipeline_state = self.pipeline_init(q, qd)

        # --- MOCAP MAGIC: Snap walls above/below ground dynamically ---
        wall_z = jnp.where(grid.flatten() == 1, 0.5, -10.0)
        xv, yv = jnp.meshgrid(jnp.arange(self.height), jnp.arange(self.width), indexing='ij')
        wall_x = xv.flatten() * self.size_scaling
        wall_y = yv.flatten() * self.size_scaling

        wall_mocap_pos = jnp.stack([wall_x, wall_y, wall_z], axis=-1)
        goal_mocap_pos = jnp.array([[goal_pos[0], goal_pos[1], 0.01]])

        # Concatenate wall mocaps + goal mocap and push to the physics engine
        new_mocap_pos = jnp.concatenate([wall_mocap_pos, goal_mocap_pos], axis=0)
        pipeline_state = pipeline_state.replace(mocap_pos=new_mocap_pos)
        
        obs = self._get_obs(pipeline_state, grid, goal_pos)
        
        info = {
            "seed": jnp.zeros((), dtype=jnp.int32),
            "truncation": jnp.zeros(()),
            "steps": jnp.zeros((), dtype=jnp.int32),
            "grid": grid,               # Save dynamic grid for step()
            "goal_pos": goal_pos        # Save dynamic goal for step()
        }
        metrics = {
            "dist": jnp.zeros(()),
            "success": jnp.zeros(()),
            "distance_from_origin": jnp.zeros(())
        }
        
        return State(pipeline_state, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        base_action = jnp.clip(action, -1.0, 1.0) * 0.2
        full_action = jnp.concatenate([base_action, self.arm_rest, self.gripper_rest])
                
        pipeline_state = self.pipeline_step(state.pipeline_state, full_action)
        
        # Retrieve the specific maze grid and goal assigned to this environment
        grid = state.info["grid"]
        goal_pos = state.info["goal_pos"]
        
        obs = self._get_obs(pipeline_state, grid, goal_pos)
        
        dist = jnp.linalg.norm(pipeline_state.x.pos[0, :2] - goal_pos[:2])
        reward = -dist - 0.1 * jnp.sum(jnp.square(action))
        done = (dist < 0.5).astype(jnp.float32)

        steps = state.info["steps"] + 1
        info = {
            **state.info,
            "seed": state.info["seed"],
            "truncation": state.info.get("truncation", jnp.zeros(())),
            "steps": steps,
            "grid": grid,
            "goal_pos": goal_pos
        }
        metrics = {
            "dist": dist,
            "success": done,
            "distance_from_origin": jnp.linalg.norm(pipeline_state.x.pos[0, :2])
        }

        return state.replace(
            pipeline_state=pipeline_state, 
            obs=obs, 
            reward=reward, 
            done=done,
            info=info,
            metrics=metrics
        )

    def _get_obs(self, pipeline_state: base.State, grid: jax.Array, goal_pos: jax.Array) -> jax.Array:
        base_pos = pipeline_state.q[:3]
        base_vel = pipeline_state.qd[:3]
        grid_obs = grid.flatten()

        return jnp.concatenate([
            base_pos,            # 3
            base_vel,            # 3
            grid_obs,            # 121 (Global knowledge)
            goal_pos[:2],        # 2 
        ])
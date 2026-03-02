import jax
import jax.numpy as jnp
from envs.manipulation.arm_reach import ArmReach

def verify_reach_task():
    # 1. Initialize the Environment
    # The MJX backend is used for JAX-accelerated MuJoCo simulation.
    env = ArmReach(backend="mjx")
    
    # 2. JIT-compile the reset and step functions for performance
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    
    # 3. Reset the environment
    rng = jax.random.PRNGKey(42)
    state = jit_reset(rng)
    
    print("--- Initial State ---")
    print(f"Observation Shape: {state.obs.shape}")
    # ArmReach observation is 13-dim (state) + 3-dim (goal) = 16 dimensions total.
    print(f"Observation (first 5 values): {state.obs[:5]}")
    print(f"Goal: {state.info['goal']}")
    
    # 4. Perform a test step
    # Action space for ArmReach is 4-dim, corresponding to joints 1, 2, 4, and 6.
    # We provide a simple action in the range [-1, 1].
    test_action = jnp.array([0.5, -0.5, 0.2, 0.0]) 
    
    print("\n--- Performing Test Step ---")
    new_state = jit_step(state, test_action)
    
    # 5. Check Outputs
    print(f"Reward: {new_state.reward}")
    print(f"Success Metrics: {new_state.metrics}")
    
    # Verify that the robot actually moved
    # Observation index 7-9 contains the end-effector position.
    pos_diff = jnp.linalg.norm(new_state.obs[7:10] - state.obs[7:10])
    print(f"EEF Movement Distance: {pos_diff:.4f} meters")

    # 6. Basic Assertions for Correctness
    assert state.obs.shape[0] == 16, "Observation dimension mismatch for ArmReach!"
    assert not jnp.allclose(state.obs, new_state.obs), "Robot did not move after taking an action!"
    print("\nVerification Complete: Environment is working correctly.")

if __name__ == "__main__":
    verify_reach_task()

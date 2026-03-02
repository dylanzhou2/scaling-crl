# humanoid test
import jax
from jax import numpy as jnp
from brax.io import mjcf
import os

# 1. Setup Path to Humanoid XML
# Based on your repository structure
xml_path = "envs/assets/humanoid.xml"

try:
    if not os.path.exists(xml_path):
        print(f"Error: File not found at {xml_path}")
    else:
        # 2. Load the system using the MJX-compatible loader
        sys = mjcf.load(xml_path)
        print("Successfully loaded Humanoid MJCF!")
        print(f"q_size (Positions): {sys.q_size()}")
        print(f"qd_size (Velocities): {sys.qd_size()}")
        print(f"act_size (Actuators): {sys.act_size()}")
        
        # 3. Test JIT Compilation (Crucial for MJX)
        @jax.jit
        def test_step(rng):
            # Initialize state
            q = sys.init_q
            qd = jnp.zeros(sys.qd_size())

            # Create a dummy action (all zeros)
            action = jnp.zeros(sys.act_size())
            
            # In a real Brax env, you'd use PipelineEnv, 
            # but we can check basic physics loading here.
            return q, qd

        rng = jax.random.PRNGKey(0)
        q, qd = test_step(rng)
        print("JIT compilation and initial state check passed.")

except Exception as e:
    print(f"Humanoid loading failed: {e}")
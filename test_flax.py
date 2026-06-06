import jax
import jax.numpy as jnp
from jax.lib import xla_bridge
import flax
print("Flax version:", flax.__version__)
print("JAX version:", jax.__version__)
print("Platform:", jax.default_backend())

# Simulate the init step
import jax
from flax import linen as nn
import numpy as np
class Dummy(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(10)(x)
try:
    dummy = Dummy()
    params = dummy.init(jax.random.PRNGKey(0), np.ones((1, 5)))
    print("Dummy init successful")
except Exception as e:
    import traceback
    traceback.print_exc()

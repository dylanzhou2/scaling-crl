import jax
import jax.numpy as jnp
from train import Actor, SA_encoder, G_encoder

# 1. Setup dummy data
obs_dim = 20
action_size = 11
batch_size = 2
obs = jnp.zeros((batch_size, obs_dim + 3)) # state + goal
actions = jnp.zeros((batch_size, action_size))
goal = obs[:, obs_dim:]

# 2. Define a "Mini-Loss" that mimics your Critic
def loss_fn(sa_params, g_params):
    sa_repr = SA_encoder().apply(sa_params, obs[:, :obs_dim], actions)
    g_repr = G_encoder().apply(g_params, goal)
    
    # THE CULPRIT: sqrt(sum(diff^2))
    # If this returns NaN gradients, you need the +1e-8 epsilon.
    dist = jnp.sqrt(jnp.sum((sa_repr - g_repr)**2, axis=-1))
    return jnp.mean(dist)

# 3. Check Gradients
sa_params = SA_encoder().init(jax.random.PRNGKey(0), obs[:, :obs_dim], actions)
g_params = G_encoder().init(jax.random.PRNGKey(1), goal)

grads = jax.grad(loss_fn)(sa_params, g_params)
print(f"Contains NaNs: {jnp.isnan(jax.flatten_util.ravel_pytree(grads)[0]).any()}")

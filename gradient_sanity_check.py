import jax
import jax.numpy as jnp

def test_gradient_stability():
    # Simulate sa_repr and g_repr being identical (distance = 0)
    sa_repr = jnp.array([[1.0, 2.0, 3.0]])
    g_repr = jnp.array([[1.0, 2.0, 3.0]])

    def loss_broken(x, y):
        return -jnp.sqrt(jnp.sum((x - y) ** 2, axis=-1)).mean()

    def loss_fixed(x, y):
        # The 1e-8 epsilon prevents the derivative (1/2sqrt(x)) from hitting 1/0
        return -jnp.sqrt(jnp.sum((x - y) ** 2, axis=-1) + 1e-8).mean()

    grad_broken = jax.grad(loss_broken)(sa_repr, g_repr)
    grad_fixed = jax.grad(loss_fixed)(sa_repr, g_repr)

    print(f"Broken Grad (Epsilon=0): {grad_broken}") # Likely [nan, nan, nan]
    print(f"Fixed Grad (Epsilon=1e-8): {grad_fixed}") # Should be [0, 0, 0]

if __name__ == "__main__":
    test_gradient_stability()
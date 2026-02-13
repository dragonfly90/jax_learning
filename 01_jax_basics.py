"""
JAX Basics: Arrays, JIT compilation, and Automatic Differentiation
==================================================================

Key concepts:
- jax.numpy: Drop-in replacement for NumPy that runs on GPU/TPU
- jit: Just-in-time compilation via XLA for speed
- grad: Automatic differentiation for computing gradients
- JAX arrays are immutable (no in-place mutation)
"""

import jax
import jax.numpy as jnp

# ---- 1. JAX Arrays (DeviceArray) ----
# JAX arrays work like NumPy but are immutable and can live on GPU/TPU

x = jnp.array([1.0, 2.0, 3.0])
y = jnp.ones((3, 3))
z = jnp.dot(y, x)
print("Dot product:", z)

# Immutability: you cannot do x[0] = 10.0
# Instead, use functional updates:
x_updated = x.at[0].set(10.0)
print("Original x:", x)
print("Updated x:", x_updated)


# ---- 2. Random Numbers ----
# JAX uses explicit PRNG keys (no global state like NumPy)

key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
random_vector = jax.random.normal(key1, shape=(5,))
print("Random vector:", random_vector)


# ---- 3. JIT Compilation ----
# jax.jit compiles a function with XLA for faster execution

def slow_fn(x):
    for _ in range(5):
        x = x @ x
    return x

fast_fn = jax.jit(slow_fn)

mat = jax.random.normal(key2, (100, 100))
# First call compiles (slower), subsequent calls are fast
result = fast_fn(mat)
print("JIT result shape:", result.shape)

# Can also use as a decorator:
@jax.jit
def another_fast_fn(x, y):
    return jnp.sin(x) + jnp.cos(y)


# ---- 4. Automatic Differentiation with grad ----
# jax.grad computes gradients of scalar-valued functions

def loss_fn(w, x, y):
    """Simple MSE loss."""
    pred = jnp.dot(x, w)
    return jnp.mean((pred - y) ** 2)

key3, key4 = jax.random.split(key1)
w = jax.random.normal(key3, (3,))
x = jax.random.normal(key4, (10, 3))
y = jnp.ones(10)

# Gradient with respect to the first argument (w)
grad_fn = jax.grad(loss_fn)
grads = grad_fn(w, x, y)
print("Gradients shape:", grads.shape)

# value_and_grad returns both the loss value and the gradient
val_grad_fn = jax.value_and_grad(loss_fn)
loss_val, grads = val_grad_fn(w, x, y)
print(f"Loss: {loss_val:.4f}, Grad norm: {jnp.linalg.norm(grads):.4f}")


# ---- 5. Higher-order Differentiation ----
# You can compose grad to get higher-order derivatives

def f(x):
    return jnp.sin(x)

df = jax.grad(f)       # first derivative: cos(x)
ddf = jax.grad(df)     # second derivative: -sin(x)

x0 = 1.0
print(f"f(x)={f(x0):.4f}, f'(x)={df(x0):.4f}, f''(x)={ddf(x0):.4f}")


# ---- 6. Jacobians and Hessians ----

def vector_fn(x):
    return jnp.array([x[0] ** 2 + x[1], x[0] * x[1] ** 3])

# Full Jacobian matrix
jacobian = jax.jacobian(vector_fn)(jnp.array([1.0, 2.0]))
print("Jacobian:\n", jacobian)

# Hessian of a scalar function
def scalar_fn(x):
    return jnp.sum(x ** 3)

hessian = jax.hessian(scalar_fn)(jnp.array([1.0, 2.0, 3.0]))
print("Hessian:\n", hessian)

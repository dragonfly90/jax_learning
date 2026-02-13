"""
Custom Derivatives and Advanced Autodiff
==========================================

Key concepts:
- jax.custom_jvp: Define custom forward-mode derivatives.
- jax.custom_vjp: Define custom reverse-mode derivatives (backprop).
- stop_gradient: Prevent gradients from flowing through a computation.
- Useful for: numerical stability, straight-through estimators,
  custom backprop rules, and interfacing with non-JAX code.
"""

import jax
import jax.numpy as jnp
from functools import partial

# ---- 1. custom_jvp: Custom Forward-Mode Derivative ----
# Use when you want to override how JAX differentiates a function

@jax.custom_jvp
def safe_log(x):
    """log(x) that handles x=0 gracefully."""
    return jnp.log(x)

@safe_log.defjvp
def safe_log_jvp(primals, tangents):
    """Custom JVP: d/dx log(x) = 1/x, but clip for numerical stability."""
    x, = primals
    x_dot, = tangents
    primal_out = safe_log(x)
    # Clip the gradient to avoid inf when x is near 0
    tangent_out = x_dot / jnp.maximum(x, 1e-6)
    return primal_out, tangent_out

# Standard log would give inf gradient at x=0
grad_safe = jax.grad(safe_log)(0.0001)
print(f"safe_log grad at 0.0001: {grad_safe:.2f}")


# ---- 2. custom_vjp: Custom Reverse-Mode Derivative ----
# More common in deep learning (backpropagation)

@jax.custom_vjp
def straight_through_relu(x):
    """ReLU with straight-through estimator gradient."""
    return jnp.maximum(x, 0)

def straight_through_relu_fwd(x):
    return straight_through_relu(x), x  # save x as "residual" for backward

def straight_through_relu_bwd(x, g):
    """Straight-through: pass gradient through regardless of sign."""
    # Normal ReLU would zero out gradient for x < 0
    # Straight-through passes it through
    return (g,)

straight_through_relu.defvjp(straight_through_relu_fwd, straight_through_relu_bwd)

# Compare gradients:
normal_relu_grad = jax.grad(lambda x: jnp.sum(jax.nn.relu(x)))(jnp.array([-1.0, 0.5, -0.5, 1.0]))
st_relu_grad = jax.grad(lambda x: jnp.sum(straight_through_relu(x)))(jnp.array([-1.0, 0.5, -0.5, 1.0]))
print(f"\nNormal ReLU grad: {normal_relu_grad}")
print(f"Straight-through grad: {st_relu_grad}")


# ---- 3. stop_gradient ----
# Prevents gradients from flowing through a sub-expression

def loss_with_target_net(params, target_params, x):
    """
    DQN-style loss where target network gradients are stopped.
    Only params receives gradients, not target_params.
    """
    pred = jnp.dot(x, params)
    # stop_gradient: treat target as a constant during backprop
    target = jax.lax.stop_gradient(jnp.dot(x, target_params))
    return jnp.mean((pred - target) ** 2)

key = jax.random.PRNGKey(0)
params = jax.random.normal(key, (3,))
target_params = jax.random.normal(key, (3,))
x = jax.random.normal(key, (5, 3))

grad_params = jax.grad(loss_with_target_net)(params, target_params, x)
print(f"\nGradient w.r.t params: {grad_params}")


# ---- 4. Gradient clipping via custom_vjp ----

@jax.custom_vjp
def clip_gradient(x):
    return x

def clip_gradient_fwd(x):
    return x, ()

def clip_gradient_bwd(_, g):
    return (jnp.clip(g, -1.0, 1.0),)

clip_gradient.defvjp(clip_gradient_fwd, clip_gradient_bwd)

def unstable_fn(x):
    return clip_gradient(x ** 3)

# Without clipping, grad at x=100 would be 30000
# With clipping, it's clamped to 1.0
print(f"\nClipped gradient at x=100: {jax.grad(unstable_fn)(100.0)}")


# ---- 5. Practical: Custom softmax with log-sum-exp trick ----

@jax.custom_jvp
def stable_softmax(x):
    """Numerically stable softmax."""
    x_max = jnp.max(x, axis=-1, keepdims=True)
    exp_x = jnp.exp(x - x_max)
    return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)

@stable_softmax.defjvp
def stable_softmax_jvp(primals, tangents):
    x, = primals
    x_dot, = tangents
    s = stable_softmax(x)
    # Jacobian-vector product for softmax: s * (x_dot - <s, x_dot>)
    tangent_out = s * (x_dot - jnp.sum(s * x_dot, axis=-1, keepdims=True))
    return s, tangent_out

logits = jnp.array([1.0, 2.0, 3.0, 100.0])  # extreme value
probs = stable_softmax(logits)
grad = jax.jacobian(stable_softmax)(logits)
print(f"\nStable softmax: {probs}")
print(f"Jacobian diagonal: {jnp.diag(grad)}")


# ---- 6. Checkpointing (recomputation to save memory) ----
# jax.checkpoint recomputes forward pass during backward to save memory

@jax.checkpoint
def heavy_layer(x):
    """This layer's activations won't be stored — recomputed during backprop."""
    x = jnp.sin(x)
    x = jnp.cos(x)
    x = jnp.tanh(x)
    return x

def model(x):
    for _ in range(10):
        x = heavy_layer(x)
    return jnp.sum(x)

x = jax.random.normal(key, (100,))
grad = jax.grad(model)(x)
print(f"\nCheckpointed gradient shape: {grad.shape}")

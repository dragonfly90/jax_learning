"""
jax.value_and_grad: Computing Values and Gradients Together
============================================================

Key concepts:
- jax.value_and_grad returns BOTH the function output and its gradient.
- More efficient than calling jax.grad separately (one forward+backward pass).
- argnums controls which arguments to differentiate w.r.t.
- has_aux=True lets you return extra outputs (metrics, intermediates).
- This is the workhorse of every JAX training loop.

Comparison:
  jax.grad(f)(x)            -> gradient only (discards function value)
  jax.value_and_grad(f)(x)  -> (value, gradient) in one pass
"""

import jax
import jax.numpy as jnp


# ---- 1. Basic Usage ----
# value_and_grad wraps a function to return (output, gradient)

def f(x):
    return jnp.sum(x ** 2)

# jax.grad — returns only the gradient
grad_fn = jax.grad(f)
print("=== 1. Basic Usage ===")
print(f"grad only:      {grad_fn(jnp.array([1.0, 2.0, 3.0]))}")

# jax.value_and_grad — returns (value, gradient) together
val_grad_fn = jax.value_and_grad(f)
value, grad = val_grad_fn(jnp.array([1.0, 2.0, 3.0]))
print(f"value:          {value}")        # 1 + 4 + 9 = 14.0
print(f"gradient:       {grad}")         # [2, 4, 6]


# ---- 2. argnums — Choosing Which Arguments to Differentiate ----
# By default, differentiates w.r.t. argument 0.

def loss_fn(w, b, x, y):
    pred = x @ w + b
    return jnp.mean((pred - y) ** 2)

key = jax.random.PRNGKey(0)
w = jax.random.normal(key, (4, 2))
b = jnp.zeros(2)
x = jax.random.normal(key, (8, 4))
y = jax.random.normal(key, (8, 2))

print("\n=== 2. argnums ===")

# Differentiate w.r.t. arg 0 (w) only — default
loss, dw = jax.value_and_grad(loss_fn)(w, b, x, y)
print(f"argnums=0 (default): loss={loss:.4f}, dw shape={dw.shape}")

# Differentiate w.r.t. arg 1 (b) only
loss, db = jax.value_and_grad(loss_fn, argnums=1)(w, b, x, y)
print(f"argnums=1:           loss={loss:.4f}, db shape={db.shape}")

# Differentiate w.r.t. both w and b — returns tuple of gradients
loss, (dw, db) = jax.value_and_grad(loss_fn, argnums=(0, 1))(w, b, x, y)
print(f"argnums=(0,1):       loss={loss:.4f}, dw shape={dw.shape}, db shape={db.shape}")


# ---- 3. has_aux — Returning Extra Outputs ----
# Sometimes you want to return metrics, logits, or intermediates alongside the loss.
# has_aux=True tells JAX: the function returns (loss, aux), differentiate only the loss.

def loss_with_metrics(params, x, y):
    pred = x @ params['w'] + params['b']
    loss = jnp.mean((pred - y) ** 2)
    # Return extra info that we don't differentiate
    metrics = {
        'mse': loss,
        'pred_mean': jnp.mean(pred),
        'pred_std': jnp.std(pred),
    }
    return loss, metrics  # (differentiable, auxiliary)

params = {'w': w, 'b': b}

print("\n=== 3. has_aux=True ===")

# Without has_aux: JAX would try to differentiate the metrics dict (error!)
# With has_aux: JAX differentiates only the first output
(loss, metrics), grads = jax.value_and_grad(loss_with_metrics, has_aux=True)(params, x, y)
print(f"loss:      {loss:.4f}")
print(f"pred_mean: {metrics['pred_mean']:.4f}")
print(f"pred_std:  {metrics['pred_std']:.4f}")
print(f"grad keys: {list(grads.keys())}")  # same structure as params


# ---- 4. With Pytrees (Nested Params) ----
# value_and_grad works naturally with pytrees — gradients have the same structure.

def mlp_loss(params, x, y):
    # Two-layer MLP
    h = jnp.tanh(x @ params['layer1']['w'] + params['layer1']['b'])
    pred = h @ params['layer2']['w'] + params['layer2']['b']
    return jnp.mean((pred - y) ** 2)

params = {
    'layer1': {'w': jax.random.normal(key, (4, 8)), 'b': jnp.zeros(8)},
    'layer2': {'w': jax.random.normal(key, (8, 2)), 'b': jnp.zeros(2)},
}

print("\n=== 4. Pytree Gradients ===")
loss, grads = jax.value_and_grad(mlp_loss)(params, x, y)
print(f"loss: {loss:.4f}")
print(f"grad structure:")
for layer_name, layer_grads in grads.items():
    for param_name, g in layer_grads.items():
        print(f"  {layer_name}/{param_name}: shape={g.shape}, norm={jnp.linalg.norm(g):.4f}")


# ---- 5. Combining with jax.jit ----
# value_and_grad composes with jit for fast compiled training steps.

import optax

print("\n=== 5. JIT + value_and_grad Training Loop ===")

optimizer = optax.adam(1e-2)
opt_state = optimizer.init(params)

@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(mlp_loss)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

for step in range(200):
    params, opt_state, loss = train_step(params, opt_state, x, y)
    if step % 40 == 0:
        print(f"  Step {step:3d}: loss={loss:.4f}")


# ---- 6. JIT + has_aux — The Complete Training Pattern ----
# The most common pattern in real JAX training code.

print("\n=== 6. Complete Training Pattern ===")

def full_loss_fn(params, x, y):
    h = jnp.tanh(x @ params['layer1']['w'] + params['layer1']['b'])
    logits = h @ params['layer2']['w'] + params['layer2']['b']
    loss = jnp.mean((logits - y) ** 2)
    return loss, {'logits': logits, 'loss': loss}

# Re-init
params = {
    'layer1': {'w': jax.random.normal(key, (4, 8)), 'b': jnp.zeros(8)},
    'layer2': {'w': jax.random.normal(key, (8, 2)), 'b': jnp.zeros(2)},
}
opt_state = optimizer.init(params)

@jax.jit
def train_step_full(params, opt_state, x, y):
    (loss, aux), grads = jax.value_and_grad(full_loss_fn, has_aux=True)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux

for step in range(200):
    params, opt_state, loss, aux = train_step_full(params, opt_state, x, y)
    if step % 40 == 0:
        print(f"  Step {step:3d}: loss={loss:.4f}, pred_mean={jnp.mean(aux['logits']):.4f}")


# ---- 7. Higher-Order: Hessian-Vector Products ----
# value_and_grad composes with other transforms like grad (for second derivatives).

print("\n=== 7. Higher-Order Derivatives ===")

def scalar_fn(x):
    return jnp.sum(jnp.sin(x) * x ** 2)

x = jnp.array([1.0, 2.0, 3.0])

# First derivative
val, grad1 = jax.value_and_grad(scalar_fn)(x)
print(f"f(x)  = {val:.4f}")
print(f"f'(x) = {grad1}")

# Second derivative (Hessian diagonal) via composing grad
hessian_diag_fn = jax.value_and_grad(lambda x: jnp.sum(jax.grad(scalar_fn)(x) ** 2))
val2, grad2 = hessian_diag_fn(x)
print(f"||f'(x)||^2 = {val2:.4f}")
print(f"gradient of ||f'(x)||^2 = {grad2}")


# ---- 8. Summary ----

print("""
=== Summary ===

jax.value_and_grad(fn)                    -> (value, grad_arg0)
jax.value_and_grad(fn, argnums=1)         -> (value, grad_arg1)
jax.value_and_grad(fn, argnums=(0,1))     -> (value, (grad_arg0, grad_arg1))
jax.value_and_grad(fn, has_aux=True)      -> ((value, aux), grad)

Common training pattern:
  @jax.jit
  def train_step(params, opt_state, x, y):
      (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, x, y)
      updates, opt_state = optimizer.update(grads, opt_state, params)
      params = optax.apply_updates(params, updates)
      return params, opt_state, loss, aux

Key points:
- Always prefer value_and_grad over separate grad + function call
- has_aux=True for returning metrics/intermediates alongside loss
- Gradients match the pytree structure of the differentiated argument
- Composes naturally with jit, vmap, and other JAX transforms
""")

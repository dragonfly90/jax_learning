"""
Scan and Control Flow: Loops and Conditionals in JAX
=====================================================

Key concepts:
- jax.lax.scan: Efficient loop primitive (like fold/reduce with carry).
- jax.lax.cond: Conditional execution (if/else).
- jax.lax.while_loop: While loops compatible with jit.
- jax.lax.fori_loop: For loops compatible with jit.
- Python control flow doesn't work inside jit — use these instead.
"""

import jax
import jax.numpy as jnp

# ---- 1. Why not Python loops inside jit? ----
# Python for/if are traced once at compile time — they can't depend on values.

# This WORKS but unrolls the loop (slow compilation for large N):
@jax.jit
def python_loop(x):
    for i in range(5):  # unrolled at trace time
        x = x + 1.0
    return x

print("Python loop:", python_loop(0.0))  # 5.0


# ---- 2. jax.lax.scan — The workhorse ----
# scan(f, carry, xs) applies f sequentially, threading a carry state.
# f(carry, x) -> (new_carry, output)

def scan_step(carry, x):
    """Running sum."""
    total = carry + x
    return total, total  # (new_carry, output_to_stack)

xs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
final_carry, all_sums = jax.lax.scan(scan_step, 0.0, xs)
print(f"\nScan running sum: {all_sums}")        # [1, 3, 6, 10, 15]
print(f"Final carry: {final_carry}")             # 15.0


# ---- 3. Scan for RNN-style computation ----

def rnn_step(h, x):
    """Simple RNN cell: h_new = tanh(W_h @ h + W_x @ x)."""
    W_h = jnp.eye(4) * 0.5
    W_x = jnp.ones((4, 3)) * 0.1
    h_new = jnp.tanh(W_h @ h + W_x @ x)
    return h_new, h_new

# Process a sequence of length 10, feature dim 3
key = jax.random.PRNGKey(0)
sequence = jax.random.normal(key, (10, 3))
h0 = jnp.zeros(4)

final_h, all_h = jax.lax.scan(rnn_step, h0, sequence)
print(f"\nRNN output shape: {all_h.shape}")  # (10, 4)
print(f"Final hidden state: {final_h}")


# ---- 4. Scan for training loops ----
# Avoids Python loop overhead, compiles the entire training as one XLA program

def make_train_step(loss_fn):
    @jax.jit
    def scan_train(params, data):
        def step(params, batch):
            x, y = batch
            loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
            new_params = jax.tree.map(lambda p, g: p - 0.01 * g, params, grads)
            return new_params, loss
        return jax.lax.scan(step, params, data)
    return scan_train

def mse_loss(params, x, y):
    pred = x @ params['w'] + params['b']
    return jnp.mean((pred - y) ** 2)

params = {'w': jnp.zeros((3, 1)), 'b': jnp.float32(0.0)}
# Simulate 20 training steps with batched data
X_batches = jax.random.normal(key, (20, 8, 3))
Y_batches = jax.random.normal(key, (20, 8, 1))

train_fn = make_train_step(mse_loss)
final_params, losses = train_fn(params, (X_batches, Y_batches))
print(f"\nTraining losses (first 5): {losses[:5]}")
print(f"Training losses (last 5): {losses[-5:]}")


# ---- 5. jax.lax.cond — Conditional execution ----
# cond(pred, true_fn, false_fn, *operands)

@jax.jit
def abs_value(x):
    return jax.lax.cond(
        x >= 0,
        lambda x: x,       # true branch
        lambda x: -x,      # false branch
        x
    )

print(f"\nabs(3.0) = {abs_value(3.0)}")
print(f"abs(-5.0) = {abs_value(-5.0)}")


# ---- 6. jax.lax.while_loop ----
# while_loop(cond_fn, body_fn, init_val)

@jax.jit
def newton_sqrt(x):
    """Compute sqrt(x) via Newton's method."""
    def cond_fn(state):
        guess, _ = state
        return jnp.abs(guess * guess - x) > 1e-6

    def body_fn(state):
        guess, i = state
        guess = (guess + x / guess) / 2.0
        return (guess, i + 1)

    guess, n_iters = jax.lax.while_loop(cond_fn, body_fn, (x / 2.0, 0))
    return guess, n_iters

result, iters = newton_sqrt(2.0)
print(f"\nsqrt(2) = {result:.6f} (in {iters} iterations)")


# ---- 7. jax.lax.fori_loop ----
# fori_loop(lower, upper, body_fn, init_val)

@jax.jit
def power(base, n):
    """Compute base^n using fori_loop."""
    def body(i, val):
        return val * base
    return jax.lax.fori_loop(0, n, body, 1.0)

print(f"\n2^10 = {power(2.0, 10)}")
print(f"3^5 = {power(3.0, 5)}")


# ---- 8. jax.lax.switch — Multi-way branch ----
# switch(index, branches, *operands)

@jax.jit
def activation(x, kind):
    """Apply different activations based on kind index."""
    return jax.lax.switch(
        kind,
        [
            lambda x: x,                    # 0: identity
            lambda x: jax.nn.relu(x),        # 1: ReLU
            lambda x: jnp.tanh(x),           # 2: tanh
            lambda x: jax.nn.sigmoid(x),     # 3: sigmoid
        ],
        x
    )

x = jnp.array([-1.0, 0.0, 1.0])
for i, name in enumerate(['identity', 'relu', 'tanh', 'sigmoid']):
    print(f"  {name}({x}) = {activation(x, i)}")

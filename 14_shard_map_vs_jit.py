"""
shard_map vs jax.jit: Explicit vs Automatic Parallelism
========================================================

Key concepts:
- shard_map: Low-level API — you write per-shard code and handle collectives
  (psum, pall_to_all, etc.) explicitly. Full control over communication.
- jax.jit + NamedSharding: High-level API — you shard the data, JIT figures
  out the communication automatically via the XLA compiler.
- Both use Mesh and PartitionSpec to describe device layout.
- shard_map is useful for custom communication patterns (e.g., ring allreduce,
  pipeline parallelism). jax.jit is preferred for most standard patterns.

Note: Run with multiple simulated devices:
  XLA_FLAGS='--xla_force_host_platform_device_count=8' python 14_shard_map_vs_jit.py
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental.shard_map import shard_map
from functools import partial
import numpy as np
import os

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

devices = jax.devices()
mesh = Mesh(np.array(devices), axis_names=('dp',))
print(f"Devices: {len(devices)}, Mesh: {mesh.shape}")


# ---- 1. Basic Example: Element-wise Operation ----
# The simplest case — no communication needed between devices.

print("\n=== 1. Element-wise Operation ===")

x = jnp.arange(32.0).reshape(8, 4)  # 8 rows, 4 cols

# --- shard_map version ---
# The function receives a LOCAL shard (1 row per device on 8 devices)
@shard_map(mesh, in_specs=(P('dp', None),), out_specs=P('dp', None))
def elementwise_shard(x_shard):
    # x_shard shape: (1, 4) — each device sees only its local piece
    return jnp.sin(x_shard) + jnp.cos(x_shard)

result_shard = elementwise_shard(x)
print(f"shard_map result shape: {result_shard.shape}")

# --- jax.jit version ---
# Same function, but JIT handles sharding automatically
@jax.jit
def elementwise_jit(x):
    # x shape: (8, 4) — you write global-view code
    return jnp.sin(x) + jnp.cos(x)

x_sharded = jax.device_put(x, NamedSharding(mesh, P('dp', None)))
result_jit = elementwise_jit(x_sharded)
print(f"jax.jit  result shape: {result_jit.shape}")
print(f"Results match: {jnp.allclose(result_shard, result_jit)}")


# ---- 2. AllReduce: Sum Across Devices ----
# This is where the difference matters — shard_map requires explicit psum.

print("\n=== 2. AllReduce (Sum Across Devices) ===")

x = jax.random.normal(jax.random.PRNGKey(0), (8, 4))

# --- shard_map version ---
# You must call lax.psum explicitly to sum across devices
@shard_map(mesh, in_specs=(P('dp', None),), out_specs=P(None, None))
def mean_shard(x_shard):
    # x_shard: (1, 4) per device
    local_sum = jnp.sum(x_shard, axis=0, keepdims=True)  # (1, 4)
    # Explicit allreduce: sum across all devices in the 'dp' axis
    global_sum = jax.lax.psum(local_sum, axis_name='dp')  # (1, 4)
    return global_sum / 8.0  # manual mean

result_shard = mean_shard(x)
print(f"shard_map mean: {result_shard[0, :3]}")

# --- jax.jit version ---
# Just write jnp.mean — JIT inserts the allreduce automatically
@jax.jit
def mean_jit(x):
    return jnp.mean(x, axis=0, keepdims=True)

x_sharded = jax.device_put(x, NamedSharding(mesh, P('dp', None)))
result_jit = mean_jit(x_sharded)
print(f"jax.jit  mean: {result_jit[0, :3]}")
print(f"Results match: {jnp.allclose(result_shard, result_jit)}")


# ---- 3. Data-Parallel Gradient Computation ----
# The most common use case: compute gradients on data shards, then allreduce.

print("\n=== 3. Data-Parallel Gradient Computation ===")

# Simple linear model: y = x @ w + b
key = jax.random.PRNGKey(42)
w = jax.random.normal(key, (4, 2))
b = jnp.zeros(2)
x_data = jax.random.normal(key, (8, 4))
y_data = x_data @ w + 0.1 * jax.random.normal(key, (8, 2))

def loss_fn(w, b, x, y):
    pred = x @ w + b
    return jnp.mean((pred - y) ** 2)

# --- shard_map version ---
# Each device computes local gradients, then we allreduce
@shard_map(
    mesh,
    in_specs=(P(None, None), P(None,), P('dp', None), P('dp', None)),
    out_specs=(P(None, None), P(None,), P()),
)
def grad_step_shard(w, b, x_shard, y_shard):
    # Each device has 1/8 of the data
    local_loss, (dw, db) = jax.value_and_grad(loss_fn, argnums=(0, 1))(
        w, b, x_shard, y_shard
    )
    # Allreduce: average gradients across all devices
    dw = jax.lax.pmean(dw, axis_name='dp')
    db = jax.lax.pmean(db, axis_name='dp')
    loss = jax.lax.pmean(local_loss, axis_name='dp')
    return dw, db, loss

dw_shard, db_shard, loss_shard = grad_step_shard(w, b, x_data, y_data)
print(f"shard_map — loss: {loss_shard:.4f}, grad_w norm: {jnp.linalg.norm(dw_shard):.4f}")

# --- jax.jit version ---
# Same result, but JIT handles everything
@jax.jit
def grad_step_jit(w, b, x, y):
    loss, (dw, db) = jax.value_and_grad(loss_fn, argnums=(0, 1))(w, b, x, y)
    return dw, db, loss

w_s = jax.device_put(w, NamedSharding(mesh, P(None, None)))
b_s = jax.device_put(b, NamedSharding(mesh, P(None,)))
x_s = jax.device_put(x_data, NamedSharding(mesh, P('dp', None)))
y_s = jax.device_put(y_data, NamedSharding(mesh, P('dp', None)))

dw_jit, db_jit, loss_jit = grad_step_jit(w_s, b_s, x_s, y_s)
print(f"jax.jit  — loss: {loss_jit:.4f}, grad_w norm: {jnp.linalg.norm(dw_jit):.4f}")
print(f"Results match: {jnp.allclose(dw_shard, dw_jit, atol=1e-5)}")


# ---- 4. Full Training Loop Comparison ----
# Side-by-side training loops using both approaches.

print("\n=== 4. Training Loop — shard_map ===")

w_sm = w.copy()
b_sm = b.copy()
lr = 0.01

@shard_map(
    mesh,
    in_specs=(P(None, None), P(None,), P('dp', None), P('dp', None)),
    out_specs=(P(None, None), P(None,), P()),
)
def train_step_shard(w, b, x_shard, y_shard):
    loss, (dw, db) = jax.value_and_grad(loss_fn, argnums=(0, 1))(
        w, b, x_shard, y_shard
    )
    dw = jax.lax.pmean(dw, axis_name='dp')
    db = jax.lax.pmean(db, axis_name='dp')
    loss = jax.lax.pmean(loss, axis_name='dp')
    w = w - lr * dw
    b = b - lr * db
    return w, b, loss

for step in range(5):
    w_sm, b_sm, loss_val = train_step_shard(w_sm, b_sm, x_data, y_data)
    print(f"  Step {step}: loss={loss_val:.4f}")


print("\n=== 4. Training Loop — jax.jit ===")

w_jt = jax.device_put(w.copy(), NamedSharding(mesh, P(None, None)))
b_jt = jax.device_put(b.copy(), NamedSharding(mesh, P(None,)))

@jax.jit
def train_step_jit(w, b, x, y):
    loss, (dw, db) = jax.value_and_grad(loss_fn, argnums=(0, 1))(w, b, x, y)
    w = w - lr * dw
    b = b - lr * db
    return w, b, loss

for step in range(5):
    w_jt, b_jt, loss_val = train_step_jit(w_jt, b_jt, x_s, y_s)
    print(f"  Step {step}: loss={loss_val:.4f}")


# ---- 5. Custom Communication: Ring AllReduce (shard_map only) ----
# shard_map shines when you need communication patterns JIT can't infer.

print("\n=== 5. Custom Communication — Ring Permute (shard_map only) ===")

# ppermute: each device sends its shard to the next device in a ring
@shard_map(mesh, in_specs=(P('dp',),), out_specs=P('dp',))
def ring_shift(x_shard):
    # Shift each shard one device to the right (ring topology)
    n = jax.lax.axis_size('dp')
    perm = [(i, (i + 1) % n) for i in range(n)]
    return jax.lax.ppermute(x_shard, axis_name='dp', perm=perm)

x = jnp.arange(8.0)
print(f"Before ring shift: {x}")
print(f"After  ring shift: {ring_shift(x)}")
# Device 0 gets value from device 7, device 1 from device 0, etc.


# ---- 6. 2D Mesh: Data + Model Parallelism ----
# shard_map can specify collectives on specific mesh axes.

print("\n=== 6. 2D Mesh with shard_map ===")

mesh_2d = Mesh(np.array(devices).reshape(4, 2), axis_names=('dp', 'mp'))

x = jax.random.normal(key, (4, 8))  # batch=4, features=8
w = jax.random.normal(key, (8, 6))  # features=8, outputs=6

# --- shard_map: shard data on dp, weights on mp ---
@shard_map(
    mesh_2d,
    in_specs=(P('dp', None), P(None, 'mp')),
    out_specs=P('dp', 'mp'),
)
def matmul_2d_shard(x_shard, w_shard):
    # x_shard: (1, 8) per dp device, w_shard: (8, 3) per mp device
    return x_shard @ w_shard

result_shard = matmul_2d_shard(x, w)
print(f"shard_map 2D result shape: {result_shard.shape}")

# --- jax.jit: same thing, automatically ---
@jax.jit
def matmul_2d_jit(x, w):
    return x @ w

x_s = jax.device_put(x, NamedSharding(mesh_2d, P('dp', None)))
w_s = jax.device_put(w, NamedSharding(mesh_2d, P(None, 'mp')))
result_jit = matmul_2d_jit(x_s, w_s)
print(f"jax.jit  2D result shape: {result_jit.shape}")
print(f"Results match: {jnp.allclose(result_shard, result_jit, atol=1e-5)}")


# ---- 7. Summary: When to Use Which ----

print("""
=== When to Use shard_map vs jax.jit ===

Use jax.jit + NamedSharding (recommended default):
  - Standard data/model/FSDP parallelism
  - Clean, high-level code (write global-view, JIT handles communication)
  - Less error-prone (no manual collectives)
  - The XLA compiler optimizes communication automatically

Use shard_map when you need:
  - Custom collective patterns (ring allreduce, pipeline stages, ppermute)
  - Fine-grained control over what each device does
  - Debugging: see exactly what runs on each shard
  - Non-standard communication that JIT can't infer
  - Expert-level optimization of communication overlap

Both use:
  - Mesh for device layout
  - PartitionSpec for describing how arrays are sharded

Key difference:
  shard_map: you write LOCAL code (per-shard) + explicit collectives
  jax.jit:   you write GLOBAL code (full-array) + automatic sharding
""")

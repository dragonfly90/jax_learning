"""
Sharding and Mesh: Modern Device Parallelism in JAX
====================================================

Key concepts:
- Mesh: A logical N-dimensional grid of devices with named axes.
- PartitionSpec (P): Describes how array dimensions map to mesh axes.
- NamedSharding: Pairs a Mesh with a PartitionSpec.
- jax.jit automatically handles distributed computation when inputs are sharded.
- This is the recommended approach (over pmap) for new code.

Note: Run with multiple simulated devices:
  XLA_FLAGS='--xla_force_host_platform_device_count=8' python 05_sharding_and_mesh.py
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import os

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

devices = jax.devices()
print(f"Devices: {len(devices)}")


# ---- 1. Creating a Mesh ----
# A Mesh is a logical grid of devices with named axes

# 1D mesh: 8 devices along axis 'data'
mesh_1d = Mesh(jax.devices(), axis_names=('data',))
print(f"1D Mesh: {mesh_1d}")

# 2D mesh: 4x2 grid with axes 'data' and 'model'
import numpy as np
devices_array = np.array(jax.devices()).reshape(4, 2)
mesh_2d = Mesh(devices_array, axis_names=('data', 'model'))
print(f"2D Mesh shape: {mesh_2d.shape}")


# ---- 2. PartitionSpec and NamedSharding ----
# PartitionSpec maps array dimensions to mesh axes

# P('data', None) means:
#   - dim 0 is sharded across 'data' axis
#   - dim 1 is replicated (not sharded)

with mesh_1d:
    # Shard a matrix: rows distributed across devices, columns replicated
    sharding_rows = NamedSharding(mesh_1d, P('data', None))

    # Fully replicated (every device has full copy)
    sharding_replicated = NamedSharding(mesh_1d, P(None, None))

    # Shard columns instead of rows
    sharding_cols = NamedSharding(mesh_1d, P(None, 'data'))

# Create a sharded array
x = jnp.arange(32).reshape(8, 4)
x_sharded = jax.device_put(x, sharding_rows)
print(f"\nSharded array shape: {x_sharded.shape}")
print(f"Sharding: {x_sharded.sharding}")


# ---- 3. Automatic Parallelism with jit ----
# When inputs are sharded, jit automatically parallelizes the computation

@jax.jit
def matmul(x, y):
    return x @ y

with mesh_1d:
    # x: sharded by rows, y: replicated
    x = jax.device_put(jnp.ones((8, 4)), NamedSharding(mesh_1d, P('data', None)))
    y = jax.device_put(jnp.ones((4, 6)), NamedSharding(mesh_1d, P(None, None)))

    result = matmul(x, y)
    print(f"\nMatmul result shape: {result.shape}")
    print(f"Result sharding: {result.sharding}")


# ---- 4. Specifying Output Sharding ----
# Use jax.jit's out_shardings to control output layout

with mesh_1d:
    out_sharding = NamedSharding(mesh_1d, P('data', None))

    @jax.jit
    def compute(x):
        return jnp.sin(x) + jnp.cos(x)

    x = jax.device_put(jnp.ones((8, 4)), NamedSharding(mesh_1d, P('data', None)))
    result = compute(x)
    print(f"\nOutput sharding: {result.sharding}")


# ---- 5. 2D Mesh: Data + Model Parallelism ----
# Common in large model training: shard data AND model weights

with mesh_2d:
    # Data: shard batch dim across 'data', replicate features
    data_sharding = NamedSharding(mesh_2d, P('data', None))

    # Weights: shard output dim across 'model', replicate input dim
    weight_sharding = NamedSharding(mesh_2d, P(None, 'model'))

    # Bias: shard across 'model' axis
    bias_sharding = NamedSharding(mesh_2d, P('model',))

    @jax.jit
    def linear_forward(x, w, b):
        return x @ w + b

    batch_size = 16
    in_features = 8
    out_features = 4

    key = jax.random.PRNGKey(0)
    x = jax.device_put(
        jax.random.normal(key, (batch_size, in_features)),
        data_sharding
    )
    w = jax.device_put(
        jax.random.normal(key, (in_features, out_features)),
        weight_sharding
    )
    b = jax.device_put(jnp.zeros(out_features), bias_sharding)

    output = linear_forward(x, w, b)
    print(f"\n2D Mesh output shape: {output.shape}")
    print(f"2D Mesh output sharding: {output.sharding}")


# ---- 6. FSDP-style Training (Fully Sharded Data Parallel) ----
# Shard both data and parameters across devices

def loss_fn(params, x, y):
    pred = x @ params['w'] + params['b']
    return jnp.mean((pred - y) ** 2)

with mesh_1d:
    param_sharding = NamedSharding(mesh_1d, P(None, None))  # replicate params
    data_sharding = NamedSharding(mesh_1d, P('data', None))

    params = {
        'w': jax.device_put(jax.random.normal(key, (4, 2)), param_sharding),
        'b': jax.device_put(jnp.zeros(2), NamedSharding(mesh_1d, P(None))),
    }

    x = jax.device_put(jax.random.normal(key, (8, 4)), data_sharding)
    y = jax.device_put(jax.random.normal(key, (8, 2)), data_sharding)

    @jax.jit
    def train_step(params, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        new_params = jax.tree.map(lambda p, g: p - 0.01 * g, params, grads)
        return new_params, loss

    for step in range(5):
        params, loss = train_step(params, x, y)
        print(f"Step {step}: loss={loss:.4f}")


# ---- 7. Visualizing Sharding ----
# jax.debug.visualize_array_sharding shows how data is distributed

with mesh_1d:
    x = jax.device_put(jnp.ones((8, 4)), NamedSharding(mesh_1d, P('data', None)))
    print("\nArray sharding visualization:")
    jax.debug.visualize_array_sharding(x)

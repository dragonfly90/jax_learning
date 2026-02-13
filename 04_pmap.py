"""
pmap: Parallel Map across Devices
==================================

Key concepts:
- pmap replicates a function across multiple devices (GPUs/TPUs).
- Each device gets a slice of the data (SPMD: Single Program Multiple Data).
- Use jax.lax.psum, pmean, pmax for cross-device communication (collectives).
- pmap is the classic approach; for newer code, consider jax.sharding (see 05).

Note: These examples will run on CPU (simulating 1 device) if no GPU/TPU
is available. Set XLA_FLAGS to simulate multiple devices:
  XLA_FLAGS='--xla_force_host_platform_device_count=4' python 04_pmap.py
"""

import jax
import jax.numpy as jnp
import os

# Simulate 4 devices on CPU (remove this if you have real GPUs/TPUs)
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

print(f"Number of devices: {jax.device_count()}")
print(f"Devices: {jax.devices()}")


# ---- 1. Basic pmap ----
# Replicate a function across devices, each getting a data shard

def square(x):
    return x ** 2

# Input shape: (num_devices, ...) — first axis is the device axis
x = jnp.arange(4.0)  # [0, 1, 2, 3] — one value per device
result = jax.pmap(square)(x)
print("pmap square:", result)  # [0, 1, 4, 9]


# ---- 2. Data-parallel training with pmap ----

def loss_fn(params, x_batch, y_batch):
    """Per-device loss: each device processes its local batch."""
    pred = jnp.dot(x_batch, params['w']) + params['b']
    return jnp.mean((pred - y_batch) ** 2)

@jax.pmap
def train_step(params, x_batch, y_batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, x_batch, y_batch)

    # Average gradients across all devices
    grads = jax.lax.pmean(grads, axis_name='devices')
    loss = jax.lax.pmean(loss, axis_name='devices')

    # SGD update (same on every device since grads are synchronized)
    new_params = jax.tree.map(lambda p, g: p - 0.01 * g, params, grads)
    return new_params, loss

# Need to specify axis_name for collectives
train_step = jax.pmap(
    lambda params, x, y: jax.value_and_grad(loss_fn)(params, x, y),
    axis_name='devices'
)

# Setup: replicate params across devices, shard data
n_devices = jax.device_count()
key = jax.random.PRNGKey(0)

params = {
    'w': jnp.zeros(3),
    'b': jnp.float32(0.0),
}
# Replicate params to all devices
replicated_params = jax.tree.map(lambda x: jnp.stack([x] * n_devices), params)
print("Replicated w shape:", replicated_params['w'].shape)  # (4, 3)

# Shard data across devices: shape (n_devices, per_device_batch, features)
total_batch = 32
per_device_batch = total_batch // n_devices
X = jax.random.normal(key, (n_devices, per_device_batch, 3))
Y = jax.random.normal(key, (n_devices, per_device_batch))

(loss, grads) = train_step(replicated_params, X, Y)
print("Per-device losses:", loss)


# ---- 3. Collectives: psum, pmean, pmax, pmin ----

@jax.pmap(axis_name='i')
def demonstrate_collectives(x):
    total = jax.lax.psum(x, axis_name='i')     # sum across devices
    mean = jax.lax.pmean(x, axis_name='i')      # mean across devices
    maximum = jax.lax.pmax(x, axis_name='i')    # max across devices
    return {'sum': total, 'mean': mean, 'max': maximum}

x = jnp.array([1.0, 2.0, 3.0, 4.0])  # one value per device
results = demonstrate_collectives(x)
print(f"\nCollectives on [1,2,3,4]:")
print(f"  psum: {results['sum']}")    # [10, 10, 10, 10]
print(f"  pmean: {results['mean']}")  # [2.5, 2.5, 2.5, 2.5]
print(f"  pmax: {results['max']}")    # [4, 4, 4, 4]


# ---- 4. pmap + vmap composition ----
# pmap across devices, vmap across batch within each device

def single_predict(w, x):
    return jnp.dot(w, x)

# Inner vmap: batch within device; Outer pmap: across devices
distributed_predict = jax.pmap(
    jax.vmap(single_predict, in_axes=(None, 0)),
    in_axes=(0, 0)
)

W = jax.random.normal(key, (n_devices, 3))           # one weight per device
X = jax.random.normal(key, (n_devices, 8, 3))        # 8 samples per device

preds = distributed_predict(W, X)
print(f"\npmap+vmap predictions shape: {preds.shape}")  # (4, 8)

"""
Distributed Multi-Host Training with JAX
==========================================

Key concepts:
- JAX multi-host: Each host (machine) runs the same Python program.
- jax.distributed.initialize() sets up communication between hosts.
- Mesh spans all devices across all hosts — sharding is global.
- Data loading must be coordinated: each host loads its own shard.
- Checkpointing with orbax for distributed saving/loading.

This file covers patterns for multi-host TPU pod training.
On a single machine, the patterns still work with simulated devices.

To run on TPU pod (e.g., v4-32 = 4 hosts x 4 chips):
  # On each host:
  python 13_distributed_training.py
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np
import os

if 'TPU_CHIPS_PER_HOST_BOUNDS' not in os.environ:
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'


# ---- 1. Multi-Host Initialization ----
# On TPU pods, call this before any JAX operation

# jax.distributed.initialize()  # Uncomment on actual multi-host setup
# This auto-discovers other hosts via environment variables:
#   - MEGASCALE_COORDINATOR_ADDRESS
#   - MEGASCALE_NUM_PROCESSES
#   - MEGASCALE_PROCESS_ID
# Or pass them explicitly:
# jax.distributed.initialize(
#     coordinator_address="host0:1234",
#     num_processes=4,
#     process_id=jax.process_index(),
# )

print("=== Distributed Setup ===")
print(f"Process {jax.process_index()} of {jax.process_count()}")
print(f"Global devices: {jax.device_count()}")
print(f"Local devices: {jax.local_device_count()}")


# ---- 2. Global Mesh Across All Hosts ----

all_devices = jax.devices()
print(f"\nAll devices: {len(all_devices)}")

# Create a global mesh spanning all hosts
# e.g., 32 chips -> (8, 4) for dp=8, mp=4
num_devices = len(all_devices)
dp_size = num_devices // 2 if num_devices > 1 else 1
mp_size = min(2, num_devices)

mesh = Mesh(
    np.array(all_devices).reshape(dp_size, mp_size),
    axis_names=('dp', 'mp')
)
print(f"Global mesh: dp={dp_size}, mp={mp_size}")


# ---- 3. Per-Host Data Loading ----
# Each host must load only its shard of the global batch

def get_local_batch(global_batch_size, seq_len, vocab_size, key):
    """
    Each host generates/loads only its portion of the global batch.
    With data parallelism, host i gets rows [i*local_batch : (i+1)*local_batch].
    """
    num_hosts = jax.process_count()
    local_batch_size = global_batch_size // num_hosts
    host_id = jax.process_index()

    # In real training, you'd use a data pipeline (tf.data, grain)
    # that skips to the correct shard:
    #   dataset.shard(num_shards=num_hosts, index=host_id)
    key = jax.random.fold_in(key, host_id)
    local_tokens = jax.random.randint(
        key, (local_batch_size, seq_len), 0, vocab_size
    )
    return local_tokens

# Each host creates its local batch
key = jax.random.PRNGKey(42)
local_batch = get_local_batch(
    global_batch_size=32, seq_len=64, vocab_size=1024, key=key
)
print(f"\nLocal batch shape: {local_batch.shape}")

# Place local data on local devices with global sharding spec
with mesh:
    global_sharding = NamedSharding(mesh, P('dp'))

    # make_array_from_single_device_arrays builds a globally-sharded array
    # from each host's local data
    local_devices = jax.local_devices()
    per_device = local_batch.shape[0] // len(local_devices)

    local_arrays = [
        jax.device_put(local_batch[i * per_device:(i + 1) * per_device], d)
        for i, d in enumerate(local_devices)
    ]

    # In multi-host, this creates a global array from local shards
    # global_batch = jax.make_array_from_single_device_arrays(
    #     shape=(32, 64),
    #     sharding=global_sharding,
    #     arrays=local_arrays,
    # )
    # For single-host demo, just use device_put:
    global_batch = jax.device_put(local_batch, global_sharding)
    print(f"Global batch sharding: {global_batch.sharding}")


# ---- 4. Distributed Training Loop Pattern ----

import flax.linen as nn
import optax

class SmallModel(nn.Module):
    vocab_size: int = 1024
    dim: int = 128

    @nn.compact
    def __call__(self, tokens):
        x = nn.Embed(self.vocab_size, self.dim)(tokens)
        x = nn.Dense(self.dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.vocab_size)(x)
        return x

model = SmallModel()
params = model.init(key, jnp.ones((1, 64), dtype=jnp.int32))['params']
optimizer = optax.adamw(1e-3)
opt_state = optimizer.init(params)

with mesh:
    # Replicate params and optimizer state
    replicated = NamedSharding(mesh, P())
    params = jax.device_put(params, replicated)
    opt_state = jax.device_put(opt_state, replicated)

    def loss_fn(params, tokens):
        logits = model.apply({'params': params}, tokens[:, :-1])
        targets = tokens[:, 1:]
        return -jnp.mean(
            jnp.take_along_axis(
                jax.nn.log_softmax(logits), targets[:, :, None], axis=-1
            )
        )

    @jax.jit
    def train_step(params, opt_state, tokens):
        loss, grads = jax.value_and_grad(loss_fn)(params, tokens)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    data_sharding = NamedSharding(mesh, P('dp'))

    print("\n=== Distributed Training ===")
    for step in range(50):
        key, subkey = jax.random.split(key)
        tokens = jax.device_put(
            jax.random.randint(subkey, (8, 64), 0, 1024),
            data_sharding
        )
        params, opt_state, loss = train_step(params, opt_state, tokens)

        if step % 10 == 0:
            print(f"Step {step}: loss={loss:.4f}")


# ---- 5. Checkpointing with Orbax ----
# orbax handles distributed checkpoint saving/loading

print("\n=== Checkpointing ===")
print("""
# pip install orbax-checkpoint

import orbax.checkpoint as ocp

# Create a checkpointer
checkpointer = ocp.StandardCheckpointer()

# Save (each host saves its local shard)
checkpointer.save(
    '/tmp/checkpoint/step_100',
    args=ocp.args.StandardSave(params),
)

# Restore with target shardings
abstract_params = jax.tree.map(
    lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
    params
)
restored = checkpointer.restore(
    '/tmp/checkpoint/step_100',
    args=ocp.args.StandardRestore(abstract_params),
)

# For async checkpointing (non-blocking):
async_checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
async_checkpointer.save(path, args=ocp.args.StandardSave(params))
async_checkpointer.wait_until_finished()  # call before program exit
""")


# ---- 6. Multi-Host Debugging Tips ----

print("=== Multi-Host Debugging ===")
print("""
1. Check all hosts see the same devices:
   print(f"Host {jax.process_index()}: {jax.device_count()} devices")

2. Verify sharding is correct:
   jax.debug.visualize_array_sharding(array)

3. Log only from host 0:
   if jax.process_index() == 0:
       print(f"Loss: {loss}")

4. Barrier synchronization:
   # Ensure all hosts reach the same point
   jax.experimental.multihost_utils.sync_global_devices("checkpoint")

5. Common errors:
   - "Mismatched mesh" — all hosts must create identical Mesh objects
   - Hanging — usually one host crashed; check all host logs
   - Shape mismatch — verify per-host data shape matches sharding
""")

"""
TPU Training with JAX
======================

Key concepts:
- JAX is designed for TPUs — most code works on TPU without changes.
- TPU pods use Mesh + NamedSharding for multi-host parallelism.
- Common strategies: data parallelism, FSDP, tensor parallelism, pipeline.
- bfloat16 is native on TPU and preferred over float16.
- TPU-specific optimizations: donate_buffers, prefetch, XLA flags.

Note: This file demonstrates patterns for TPU training. On CPU/GPU it
will run but with simulated devices. For actual TPU usage, run on
Google Cloud TPU VMs or Colab TPU runtimes.

Setup on TPU VM:
  pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import optax
import flax.linen as nn
from functools import partial
import numpy as np
import os

# Simulate 8 devices for demonstration
if 'TPU_CHIPS_PER_HOST_BOUNDS' not in os.environ:
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'


# ---- 1. TPU Device Discovery ----

print("=== Device Info ===")
print(f"Backend: {jax.default_backend()}")
print(f"Num devices: {jax.device_count()}")
print(f"Local devices: {jax.local_device_count()}")
print(f"Process index: {jax.process_index()}")
print(f"Process count: {jax.process_count()}")
for d in jax.devices()[:4]:
    print(f"  {d}")


# ---- 2. Creating Meshes for TPU Topologies ----
# TPU v4 pods have 2D/3D physical topologies; meshes map to these

devices = np.array(jax.devices())

# 1D mesh: pure data parallelism
mesh_dp = Mesh(devices, axis_names=('dp',))

# 2D mesh: data + model parallelism (e.g., 4x2 for 8 devices)
mesh_2d = Mesh(devices.reshape(4, 2), axis_names=('dp', 'mp'))

# For large TPU pods (e.g., v4-128 = 64 chips):
# mesh = Mesh(devices.reshape(16, 4), axis_names=('dp', 'mp'))
# or 3D: Mesh(devices.reshape(4, 4, 4), axis_names=('dp', 'fsdp', 'mp'))

print(f"\n2D Mesh: dp={mesh_2d.shape['dp']}, mp={mesh_2d.shape['mp']}")


# ---- 3. bfloat16 Mixed Precision ----
# TPUs have native bfloat16 support — always use it for training

class LinearBF16(nn.Module):
    features: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        # Compute in bfloat16, keep master weights in float32
        kernel = self.param(
            'kernel',
            nn.initializers.lecun_normal(),
            (x.shape[-1], self.features),
        )
        # Cast to bf16 for matmul (master weights stay fp32)
        return x.astype(self.dtype) @ kernel.astype(self.dtype)


class TransformerBlockTPU(nn.Module):
    """Transformer block optimized for TPU with bf16."""
    num_heads: int
    head_dim: int
    mlp_dim: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        embed_dim = x.shape[-1]

        # Attention
        residual = x
        x = nn.RMSNorm(dtype=self.dtype)(x)
        x = x.astype(self.dtype)

        qkv = nn.Dense(3 * self.num_heads * self.head_dim,
                        use_bias=False, dtype=self.dtype)(x)
        batch, seq_len = x.shape[:2]
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        scale = 1.0 / jnp.sqrt(self.head_dim).astype(self.dtype)
        scores = jnp.einsum('bqhd,bkhd->bhqk', q, k) * scale
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))[None, None]
        scores = jnp.where(mask, scores, jnp.finfo(self.dtype).min)
        weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(self.dtype)
        attn_out = jnp.einsum('bhqk,bkhd->bqhd', weights, v)
        attn_out = attn_out.reshape(batch, seq_len, -1)
        attn_out = nn.Dense(embed_dim, use_bias=False, dtype=self.dtype)(attn_out)
        x = residual + attn_out.astype(x.dtype)

        # MLP with SwiGLU
        residual = x
        x = nn.RMSNorm(dtype=self.dtype)(x)
        x = x.astype(self.dtype)
        gate = nn.Dense(self.mlp_dim, use_bias=False, dtype=self.dtype)(x)
        up = nn.Dense(self.mlp_dim, use_bias=False, dtype=self.dtype)(x)
        x = jax.nn.silu(gate) * up
        x = nn.Dense(embed_dim, use_bias=False, dtype=self.dtype)(x)
        x = residual + x.astype(residual.dtype)

        return x


# ---- 4. Data Parallel Training on TPU ----

class SimpleModel(nn.Module):
    vocab_size: int = 1024
    embed_dim: int = 256
    num_heads: int = 4
    head_dim: int = 64
    mlp_dim: int = 512
    num_layers: int = 2

    @nn.compact
    def __call__(self, tokens):
        x = nn.Embed(self.vocab_size, self.embed_dim)(tokens)
        for _ in range(self.num_layers):
            x = TransformerBlockTPU(
                self.num_heads, self.head_dim, self.mlp_dim
            )(x)
        x = nn.RMSNorm()(x)
        return nn.Dense(self.vocab_size, use_bias=False)(x)

model = SimpleModel()
key = jax.random.PRNGKey(0)
dummy = jnp.ones((2, 64), dtype=jnp.int32)
params = model.init(key, dummy)['params']

num_params = sum(p.size for p in jax.tree.leaves(params))
print(f"\nModel parameters: {num_params:,}")

# Shard parameters and data across the mesh
with mesh_dp:
    # Replicate params across all devices
    param_sharding = NamedSharding(mesh_dp, P())
    params = jax.device_put(params, param_sharding)

    # Shard data batch dimension across devices
    data_sharding = NamedSharding(mesh_dp, P('dp'))

    optimizer = optax.adamw(3e-4, weight_decay=0.1)
    opt_state = optimizer.init(params)


# ---- 5. Training with donate_buffers ----
# donate_buffers tells XLA it can reuse input buffers for outputs,
# reducing TPU HBM memory usage

def loss_fn(params, tokens):
    logits = model.apply({'params': params}, tokens[:, :-1])
    targets = tokens[:, 1:]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(
        log_probs, targets[:, :, None], axis=-1
    )[:, :, 0]
    return -jnp.mean(target_log_probs)

@partial(jax.jit, donate_argnames=('params', 'opt_state'))
def train_step(params, opt_state, tokens):
    loss, grads = jax.value_and_grad(loss_fn)(params, tokens)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

print("\n=== Data Parallel Training ===")
with mesh_dp:
    for step in range(100):
        key, subkey = jax.random.split(key)
        # Shard data across devices
        tokens = jax.random.randint(subkey, (8, 64), 0, 1024)
        tokens = jax.device_put(tokens, data_sharding)

        params, opt_state, loss = train_step(params, opt_state, tokens)

        if step % 20 == 0:
            print(f"Step {step}: loss={loss:.4f}")


# ---- 6. FSDP (Fully Sharded Data Parallel) ----
# Shard parameters across devices to reduce per-device memory

print("\n=== FSDP Training ===")

# Re-initialize for FSDP
params_fsdp = model.init(key, dummy)['params']

with mesh_dp:
    # Shard parameters across data parallel axis
    # Each device holds 1/N of the parameters
    def shard_params(param):
        """Shard large params, replicate small ones."""
        if param.ndim >= 2 and param.shape[0] >= jax.device_count():
            return NamedSharding(mesh_dp, P('dp', None))
        return NamedSharding(mesh_dp, P())  # replicate small params

    param_shardings = jax.tree.map(shard_params, params_fsdp)
    params_fsdp = jax.tree.map(
        lambda p, s: jax.device_put(p, s),
        params_fsdp, param_shardings
    )

    opt_state_fsdp = optimizer.init(params_fsdp)

    # FSDP train step — jit handles the all-gather/reduce-scatter
    @jax.jit
    def fsdp_train_step(params, opt_state, tokens):
        loss, grads = jax.value_and_grad(loss_fn)(params, tokens)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for step in range(50):
        key, subkey = jax.random.split(key)
        tokens = jax.device_put(
            jax.random.randint(subkey, (8, 64), 0, 1024),
            data_sharding
        )
        params_fsdp, opt_state_fsdp, loss = fsdp_train_step(
            params_fsdp, opt_state_fsdp, tokens
        )
        if step % 10 == 0:
            print(f"FSDP Step {step}: loss={loss:.4f}")


# ---- 7. Tensor Parallelism with 2D Mesh ----

print("\n=== 2D Mesh (Data + Tensor Parallel) ===")

with mesh_2d:
    # Data sharded on dp axis, model weights sharded on mp axis
    data_sharding_2d = NamedSharding(mesh_2d, P('dp'))

    # Shard weight matrices along their output dimension
    def shard_for_tp(param):
        if param.ndim == 2:
            return NamedSharding(mesh_2d, P(None, 'mp'))
        return NamedSharding(mesh_2d, P())

    params_tp = model.init(key, dummy)['params']
    tp_shardings = jax.tree.map(shard_for_tp, params_tp)
    params_tp = jax.tree.map(
        lambda p, s: jax.device_put(p, s), params_tp, tp_shardings
    )

    print(f"Mesh shape: dp={mesh_2d.shape['dp']}, mp={mesh_2d.shape['mp']}")

    @jax.jit
    def tp_train_step(params, tokens):
        loss, grads = jax.value_and_grad(loss_fn)(params, tokens)
        return grads, loss

    tokens = jax.device_put(
        jax.random.randint(key, (4, 64), 0, 1024),
        data_sharding_2d
    )
    grads, loss = tp_train_step(params_tp, tokens)
    print(f"TP forward+backward loss: {loss:.4f}")


# ---- 8. TPU Performance Tips ----

print("\n=== TPU Performance Tips ===")
print("""
1. Use bfloat16: Native on TPU, 2x throughput vs float32
   x = x.astype(jnp.bfloat16)

2. Pad batch/sequence to multiples of 128:
   TPU tiles are 128x128, so dimensions matching this are fastest

3. donate_buffers: Reuse input memory for outputs
   @partial(jax.jit, donate_argnames=('params', 'opt_state'))

4. Prefetch data: Overlap data loading with computation
   Use tf.data or grain for async data pipelines

5. Profile with TensorBoard:
   jax.profiler.start_trace('/tmp/tensorboard')
   # ... training steps ...
   jax.profiler.stop_trace()

6. Avoid Python overhead in the training loop:
   Use jax.lax.scan for the inner loop when possible

7. XLA flags for TPU optimization:
   XLA_FLAGS='--xla_tpu_megacore_fusion_allow_ags=true'

8. Check for recompilation:
   jax.config.update('jax_log_compiles', True)
""")

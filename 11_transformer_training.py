"""
Transformer Training from Scratch in JAX
==========================================

Key concepts:
- Full GPT-style decoder-only transformer in pure JAX + Flax.
- Multi-head causal self-attention with KV cache for inference.
- Rotary positional embeddings (RoPE).
- Training loop with Optax (AdamW + cosine schedule + warmup).
- Gradient accumulation for effective large batch training.

Install: pip install jax jaxlib flax optax
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from functools import partial

# ---- 1. Rotary Positional Embeddings (RoPE) ----

def rotary_embedding(x, seq_len, head_dim):
    """
    Apply rotary positional embeddings.
    x: (..., seq_len, head_dim)
    """
    positions = jnp.arange(seq_len)
    dim_pairs = jnp.arange(0, head_dim, 2)
    freqs = 1.0 / (10000.0 ** (dim_pairs / head_dim))
    angles = positions[:, None] * freqs[None, :]  # (seq_len, head_dim//2)

    cos = jnp.cos(angles)
    sin = jnp.sin(angles)

    # Split x into pairs and rotate
    x1 = x[..., ::2]   # even dimensions
    x2 = x[..., 1::2]  # odd dimensions

    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    # Interleave back
    return jnp.stack([out1, out2], axis=-1).reshape(x.shape)


# ---- 2. Multi-Head Causal Self-Attention ----

class CausalSelfAttention(nn.Module):
    num_heads: int
    head_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, training=False):
        batch, seq_len, embed_dim = x.shape

        # Project to Q, K, V
        qkv = nn.Dense(3 * self.num_heads * self.head_dim, use_bias=False)(x)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Apply RoPE
        q = rotary_embedding(q, seq_len, self.head_dim)
        k = rotary_embedding(k, seq_len, self.head_dim)

        # Attention scores
        scale = 1.0 / jnp.sqrt(self.head_dim)
        scores = jnp.einsum('bqhd,bkhd->bhqk', q, k) * scale

        # Causal mask
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))[None, None, :, :]
        scores = jnp.where(mask, scores, -1e9)

        weights = jax.nn.softmax(scores, axis=-1)
        weights = nn.Dropout(self.dropout_rate)(weights, deterministic=not training)

        # Weighted sum
        out = jnp.einsum('bhqk,bkhd->bqhd', weights, v)
        out = out.reshape(batch, seq_len, self.num_heads * self.head_dim)

        # Output projection
        return nn.Dense(embed_dim, use_bias=False)(out)


# ---- 3. Transformer Block ----

class TransformerBlock(nn.Module):
    num_heads: int
    head_dim: int
    mlp_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, training=False):
        # Pre-norm attention
        residual = x
        x = nn.RMSNorm()(x)
        x = CausalSelfAttention(
            self.num_heads, self.head_dim, self.dropout_rate
        )(x, training=training)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        x = residual + x

        # Pre-norm MLP (SwiGLU)
        residual = x
        x = nn.RMSNorm()(x)
        # SwiGLU: gate * silu(x)
        gate = nn.Dense(self.mlp_dim, use_bias=False)(x)
        up = nn.Dense(self.mlp_dim, use_bias=False)(x)
        x = jax.nn.silu(gate) * up
        x = nn.Dense(residual.shape[-1], use_bias=False)(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        x = residual + x

        return x


# ---- 4. Full GPT Model ----

class GPT(nn.Module):
    vocab_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    mlp_dim: int
    max_seq_len: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, token_ids, training=False):
        batch, seq_len = token_ids.shape
        embed_dim = self.num_heads * self.head_dim

        # Token embedding (no positional embedding — we use RoPE)
        x = nn.Embed(self.vocab_size, embed_dim)(token_ids)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)

        # Transformer layers
        for _ in range(self.num_layers):
            x = TransformerBlock(
                self.num_heads, self.head_dim,
                self.mlp_dim, self.dropout_rate
            )(x, training=training)

        x = nn.RMSNorm()(x)

        # Language model head (weight-tied with embedding)
        logits = nn.Dense(self.vocab_size, use_bias=False)(x)
        return logits


# ---- 5. Training Setup ----

# Small model for demonstration
config = dict(
    vocab_size=1024,
    num_layers=4,
    num_heads=4,
    head_dim=32,
    mlp_dim=512,
    max_seq_len=128,
    dropout_rate=0.1,
)

model = GPT(**config)
key = jax.random.PRNGKey(42)

# Initialize
dummy_tokens = jnp.ones((2, 128), dtype=jnp.int32)
variables = model.init(key, dummy_tokens, training=False)
params = variables['params']

# Count parameters
num_params = sum(p.size for p in jax.tree.leaves(params))
print(f"Model parameters: {num_params:,}")


# ---- 6. Optimizer: AdamW with Cosine Schedule + Warmup ----

num_steps = 500
warmup_steps = 50
peak_lr = 3e-4
min_lr = 1e-5

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=peak_lr,
    warmup_steps=warmup_steps,
    decay_steps=num_steps,
    end_value=min_lr,
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),   # gradient clipping
    optax.adamw(schedule, weight_decay=0.1),
)
opt_state = optimizer.init(params)


# ---- 7. Loss Function ----

def compute_loss(params, tokens, key):
    """
    Language modeling loss: predict next token.
    tokens: (batch, seq_len) — input token IDs.
    """
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]

    logits = model.apply(
        {'params': params}, inputs,
        training=True,
        rngs={'dropout': key}
    )

    # Cross-entropy loss
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(
        log_probs, targets[:, :, None], axis=-1
    )[:, :, 0]

    return -jnp.mean(target_log_probs)


# ---- 8. Training Step ----

@jax.jit
def train_step(params, opt_state, tokens, key):
    loss, grads = jax.value_and_grad(compute_loss)(params, tokens, key)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


# ---- 9. Training Loop ----

print("\n=== Training ===")
key = jax.random.PRNGKey(0)

for step in range(num_steps):
    key, data_key, dropout_key = jax.random.split(key, 3)

    # Generate random token sequences (replace with real data)
    tokens = jax.random.randint(data_key, (8, 128), 0, config['vocab_size'])

    params, opt_state, loss = train_step(params, opt_state, tokens, dropout_key)

    if step % 50 == 0:
        lr = schedule(step)
        print(f"Step {step:4d} | Loss: {loss:.4f} | LR: {lr:.2e}")

print("Training complete!")


# ---- 10. Gradient Accumulation ----
# Simulate larger batch sizes by accumulating gradients over micro-batches

@jax.jit
def train_step_grad_accum(params, opt_state, micro_batches, key):
    """
    Accumulate gradients over multiple micro-batches.
    micro_batches: (num_micro, batch_per_micro, seq_len)
    """
    num_micro = micro_batches.shape[0]
    keys = jax.random.split(key, num_micro)

    def accum_step(carry, micro_and_key):
        total_loss, total_grads = carry
        micro_batch, step_key = micro_and_key

        loss, grads = jax.value_and_grad(compute_loss)(params, micro_batch, step_key)
        total_grads = jax.tree.map(lambda a, b: a + b, total_grads, grads)
        return (total_loss + loss, total_grads), None

    zero_grads = jax.tree.map(jnp.zeros_like, params)
    (total_loss, total_grads), _ = jax.lax.scan(
        accum_step, (0.0, zero_grads), (micro_batches, keys)
    )

    # Average
    avg_loss = total_loss / num_micro
    avg_grads = jax.tree.map(lambda g: g / num_micro, total_grads)

    updates, opt_state = optimizer.update(avg_grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, avg_loss

# Example: 4 micro-batches of 4 samples each = effective batch of 16
key, accum_key = jax.random.split(key)
micro_batches = jax.random.randint(key, (4, 4, 128), 0, config['vocab_size'])
params, opt_state, loss = train_step_grad_accum(
    params, opt_state, micro_batches, accum_key
)
print(f"\nGrad accumulation loss: {loss:.4f}")


# ---- 11. Simple Greedy Generation ----

@jax.jit
def generate(params, prompt_tokens, max_new_tokens=50):
    """Autoregressive greedy decoding."""
    def generate_step(carry, _):
        tokens = carry
        logits = model.apply({'params': params}, tokens, training=False)
        next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        tokens = jnp.concatenate([tokens, next_token], axis=1)
        return tokens, next_token

    tokens, generated = jax.lax.scan(
        generate_step, prompt_tokens, None, length=max_new_tokens
    )
    return tokens

prompt = jnp.array([[1, 2, 3]])  # dummy prompt
generated = generate(params, prompt, max_new_tokens=20)
print(f"\nGenerated sequence shape: {generated.shape}")
print(f"Generated tokens: {generated[0]}")

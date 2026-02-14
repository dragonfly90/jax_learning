"""
Flash Attention in JAX
=======================

Key concepts:
- Flash Attention computes exact attention without materializing the full
  N×N attention matrix, reducing memory from O(N²) to O(N).
- It works by tiling Q, K, V into blocks and computing attention incrementally
  using the online softmax trick (log-sum-exp correction).
- JAX's splash_attention / dot_product_attention provides hardware-optimized
  implementations on TPU; this file shows both the manual algorithm and
  the built-in API.

Reference: "FlashAttention: Fast and Memory-Efficient Exact Attention
with IO-Awareness" (Dao et al., 2022)
"""

import jax
import jax.numpy as jnp
from functools import partial

# ---- 1. Standard (Naive) Attention ----
# Materializes the full N×N matrix — O(N²) memory

def naive_attention(q, k, v, mask=None):
    """
    Standard scaled dot-product attention.
    q, k, v: (batch, heads, seq_len, head_dim)
    """
    d_k = q.shape[-1]
    scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(d_k)
    if mask is not None:
        scores = jnp.where(mask, scores, -1e9)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.matmul(weights, v)


# ---- 2. Flash Attention (Manual Block-wise Implementation) ----
# Processes Q in blocks, iterates over K/V blocks, never stores full N×N

def flash_attention_manual(q, k, v, block_size=64):
    """
    Flash Attention via block-wise computation with online softmax.
    q, k, v: (seq_len, head_dim) — single head, no batch for clarity.

    Algorithm:
    1. Split Q into blocks of size Br
    2. For each Q block, iterate over all K/V blocks
    3. Compute local attention scores and accumulate output using
       the online log-sum-exp trick to avoid materializing full NxN
    """
    seq_len, d = q.shape
    num_blocks_q = seq_len // block_size
    num_blocks_kv = seq_len // block_size
    scale = 1.0 / jnp.sqrt(d)

    output = jnp.zeros_like(q)

    for i in range(num_blocks_q):
        q_block = q[i * block_size:(i + 1) * block_size]  # (Br, d)

        # Running statistics for online softmax
        m_i = jnp.full((block_size, 1), -jnp.inf)  # row-wise max
        l_i = jnp.zeros((block_size, 1))             # row-wise sum of exp
        o_i = jnp.zeros((block_size, d))              # accumulated output

        for j in range(num_blocks_kv):
            k_block = k[j * block_size:(j + 1) * block_size]  # (Bc, d)
            v_block = v[j * block_size:(j + 1) * block_size]  # (Bc, d)

            # Local attention scores: (Br, Bc)
            s_ij = (q_block @ k_block.T) * scale

            # Online softmax update
            m_ij = jnp.max(s_ij, axis=-1, keepdims=True)       # local max
            m_new = jnp.maximum(m_i, m_ij)                       # global max

            # Correction factors for previously accumulated values
            exp_old = jnp.exp(m_i - m_new)
            exp_new = jnp.exp(m_ij - m_new)

            p_ij = jnp.exp(s_ij - m_new)  # (Br, Bc) softmax numerator

            l_new = exp_old * l_i + jnp.sum(p_ij, axis=-1, keepdims=True)

            # Update output: rescale old output + add new contribution
            o_i = exp_old * o_i + p_ij @ v_block

            m_i = m_new
            l_i = l_new

        # Normalize
        o_i = o_i / l_i
        output = output.at[i * block_size:(i + 1) * block_size].set(o_i)

    return output


# ---- 3. Verify correctness ----

key = jax.random.PRNGKey(0)
seq_len, d = 256, 64

q = jax.random.normal(key, (seq_len, d))
k = jax.random.normal(key, (seq_len, d))
v = jax.random.normal(key, (seq_len, d))

naive_out = naive_attention(
    q[None, None], k[None, None], v[None, None]
)[0, 0]
flash_out = flash_attention_manual(q, k, v, block_size=64)

print(f"Max difference (naive vs flash): {jnp.max(jnp.abs(naive_out - flash_out)):.2e}")
print(f"Mean difference: {jnp.mean(jnp.abs(naive_out - flash_out)):.2e}")


# ---- 4. JIT-compiled batched Flash Attention ----

@partial(jax.jit, static_argnames=['block_size'])
def flash_attention_jit(q, k, v, block_size=64):
    """
    JIT-friendly flash attention using lax.scan over KV blocks.
    q, k, v: (seq_len, head_dim)
    """
    seq_len, d = q.shape
    scale = 1.0 / jnp.sqrt(d)
    num_blocks_kv = seq_len // block_size

    # Reshape K, V into blocks: (num_blocks, block_size, d)
    k_blocks = k.reshape(num_blocks_kv, block_size, d)
    v_blocks = v.reshape(num_blocks_kv, block_size, d)

    def process_query_block(q_block):
        """Process one Q block against all KV blocks."""
        def scan_fn(carry, kv_block):
            m_i, l_i, o_i = carry
            k_block, v_block = kv_block

            s_ij = (q_block @ k_block.T) * scale
            m_ij = jnp.max(s_ij, axis=-1, keepdims=True)
            m_new = jnp.maximum(m_i, m_ij)

            exp_old = jnp.exp(m_i - m_new)
            p_ij = jnp.exp(s_ij - m_new)
            l_new = exp_old * l_i + jnp.sum(p_ij, axis=-1, keepdims=True)
            o_i = exp_old * o_i + p_ij @ v_block

            return (m_new, l_new, o_i), None

        init = (
            jnp.full((block_size, 1), -jnp.inf),
            jnp.zeros((block_size, 1)),
            jnp.zeros((block_size, d)),
        )
        (m_final, l_final, o_final), _ = jax.lax.scan(
            scan_fn, init, (k_blocks, v_blocks)
        )
        return o_final / l_final

    # Process all Q blocks via vmap
    q_blocks = q.reshape(-1, block_size, d)
    output_blocks = jax.vmap(process_query_block)(q_blocks)
    return output_blocks.reshape(seq_len, d)

flash_jit_out = flash_attention_jit(q, k, v)
print(f"\nJIT flash vs naive max diff: {jnp.max(jnp.abs(naive_out - flash_jit_out)):.2e}")


# ---- 5. Multi-head Flash Attention ----

@partial(jax.jit, static_argnames=['block_size'])
def multi_head_flash_attention(q, k, v, block_size=64):
    """
    Multi-head flash attention.
    q, k, v: (batch, num_heads, seq_len, head_dim)
    """
    def single_head_flash(q, k, v):
        return flash_attention_jit(q, k, v, block_size=block_size)

    # vmap over heads, then over batch
    batched = jax.vmap(jax.vmap(single_head_flash))(q, k, v)
    return batched

batch, heads, seq_len, d = 2, 8, 256, 64
key1, key2, key3 = jax.random.split(key, 3)
Q = jax.random.normal(key1, (batch, heads, seq_len, d))
K = jax.random.normal(key2, (batch, heads, seq_len, d))
V = jax.random.normal(key3, (batch, heads, seq_len, d))

mha_out = multi_head_flash_attention(Q, K, V)
naive_mha = naive_attention(Q, K, V)
print(f"\nMulti-head flash attention output shape: {mha_out.shape}")
print(f"MHA max diff vs naive: {jnp.max(jnp.abs(mha_out - naive_mha)):.2e}")


# ---- 6. Causal (Autoregressive) Flash Attention ----

@partial(jax.jit, static_argnames=['block_size'])
def causal_flash_attention(q, k, v, block_size=64):
    """
    Causal flash attention — each position only attends to previous positions.
    q, k, v: (seq_len, head_dim)
    """
    seq_len, d = q.shape
    scale = 1.0 / jnp.sqrt(d)
    num_blocks = seq_len // block_size

    k_blocks = k.reshape(num_blocks, block_size, d)
    v_blocks = v.reshape(num_blocks, block_size, d)

    def process_query_block(q_idx_and_block):
        q_idx, q_block = q_idx_and_block

        def scan_fn(carry, kv_idx_and_block):
            m_i, l_i, o_i = carry
            kv_idx, k_block, v_block = kv_idx_and_block

            s_ij = (q_block @ k_block.T) * scale

            # Causal mask: Q position i can attend to K position j if j <= i
            q_positions = q_idx * block_size + jnp.arange(block_size)[:, None]
            k_positions = kv_idx * block_size + jnp.arange(block_size)[None, :]
            causal_mask = q_positions >= k_positions
            s_ij = jnp.where(causal_mask, s_ij, -1e9)

            m_ij = jnp.max(s_ij, axis=-1, keepdims=True)
            m_new = jnp.maximum(m_i, m_ij)
            exp_old = jnp.exp(m_i - m_new)
            p_ij = jnp.exp(s_ij - m_new)
            l_new = exp_old * l_i + jnp.sum(p_ij, axis=-1, keepdims=True)
            o_i = exp_old * o_i + p_ij @ v_block

            return (m_new, l_new, o_i), None

        init = (
            jnp.full((block_size, 1), -jnp.inf),
            jnp.zeros((block_size, 1)),
            jnp.zeros((block_size, d)),
        )
        kv_indices = jnp.arange(num_blocks)
        (m_f, l_f, o_f), _ = jax.lax.scan(
            scan_fn, init, (kv_indices, k_blocks, v_blocks)
        )
        return o_f / l_f

    q_blocks = q.reshape(num_blocks, block_size, d)
    q_indices = jnp.arange(num_blocks)
    output_blocks = jax.vmap(process_query_block)((q_indices, q_blocks))
    return output_blocks.reshape(seq_len, d)

# Verify causal attention
q_test = jax.random.normal(key, (256, 64))
k_test = jax.random.normal(key, (256, 64))
v_test = jax.random.normal(key, (256, 64))

causal_out = causal_flash_attention(q_test, k_test, v_test)

# Compare with naive causal
causal_mask = jnp.tril(jnp.ones((256, 256), dtype=bool))[None, None]
naive_causal = naive_attention(
    q_test[None, None], k_test[None, None], v_test[None, None], mask=causal_mask
)[0, 0]
print(f"\nCausal flash vs naive max diff: {jnp.max(jnp.abs(causal_out - naive_causal)):.2e}")

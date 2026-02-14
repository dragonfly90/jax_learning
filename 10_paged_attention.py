"""
Paged Attention in JAX
=======================

Key concepts:
- Paged Attention is used in LLM serving (vLLM) to manage KV cache memory
  efficiently during autoregressive generation.
- Instead of pre-allocating contiguous KV cache for max_seq_len, memory is
  divided into fixed-size "pages" (blocks) allocated on demand.
- A page table maps each sequence's logical positions to physical page indices.
- This eliminates memory waste from padding and enables flexible memory sharing
  (e.g., beam search candidates can share prefix pages via copy-on-write).

Reference: "Efficient Memory Management for Large Language Model Serving
with PagedAttention" (Kwon et al., 2023)
"""

import jax
import jax.numpy as jnp
from functools import partial

# ---- 1. Standard KV Cache (Wasteful) ----
# Pre-allocates [batch, max_seq_len, heads, dim] — wastes memory on padding

def standard_kv_cache_attention(q, k_cache, v_cache, seq_lens):
    """
    q: (batch, 1, heads, dim) — current query token
    k_cache, v_cache: (batch, max_seq_len, heads, dim)
    seq_lens: (batch,) — actual sequence lengths
    """
    batch, max_len, heads, dim = k_cache.shape
    scale = 1.0 / jnp.sqrt(dim)

    # (batch, heads, 1, max_len)
    scores = jnp.einsum('bqhd,bkhd->bhqk', q, k_cache) * scale

    # Mask out padding positions
    positions = jnp.arange(max_len)[None, None, None, :]  # (1,1,1,max_len)
    mask = positions < seq_lens[:, None, None, None]        # (batch,1,1,max_len)
    scores = jnp.where(mask, scores, -1e9)

    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum('bhqk,bkhd->bqhd', weights, v_cache)


# ---- 2. Paged KV Cache Data Structure ----

class PagedKVCache:
    """
    Paged KV cache for efficient LLM serving.

    Physical layout:
      k_pages: (num_pages, page_size, num_heads, head_dim)
      v_pages: (num_pages, page_size, num_heads, head_dim)

    Logical mapping:
      page_table: (batch, max_pages_per_seq) — maps logical page index
                  to physical page index for each sequence.
      seq_lens: (batch,) — actual token count per sequence.
    """
    def __init__(self, num_pages, page_size, num_heads, head_dim):
        self.page_size = page_size
        self.num_pages = num_pages
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Physical page pool (shared across all sequences)
        self.k_pages = jnp.zeros((num_pages, page_size, num_heads, head_dim))
        self.v_pages = jnp.zeros((num_pages, page_size, num_heads, head_dim))

        # Track which pages are free
        self.free_pages = list(range(num_pages))

    def allocate_page(self):
        """Get a free page index."""
        if not self.free_pages:
            raise RuntimeError("Out of KV cache pages!")
        return self.free_pages.pop(0)

    def free_page(self, page_idx):
        """Return a page to the free pool."""
        self.free_pages.append(page_idx)


# ---- 3. Paged Attention Kernel ----

@partial(jax.jit, static_argnames=['page_size'])
def paged_attention(
    q,            # (batch, 1, num_heads, head_dim) — single query token
    k_pages,      # (num_pages, page_size, num_heads, head_dim)
    v_pages,      # (num_pages, page_size, num_heads, head_dim)
    page_table,   # (batch, max_pages_per_seq) — physical page indices
    seq_lens,     # (batch,) — actual sequence lengths
    page_size=16,
):
    """
    Paged attention: look up KV from pages via the page table.

    Instead of contiguous KV cache, we gather K/V from scattered pages.
    """
    batch, _, num_heads, head_dim = q.shape
    max_pages = page_table.shape[1]
    scale = 1.0 / jnp.sqrt(head_dim)

    def single_sequence_attention(q_single, page_indices, seq_len):
        """Attention for one sequence."""
        # q_single: (1, num_heads, head_dim)
        # page_indices: (max_pages_per_seq,)

        # Gather K/V pages for this sequence
        # k_seq: (max_pages, page_size, num_heads, head_dim)
        k_seq = k_pages[page_indices]
        v_seq = v_pages[page_indices]

        # Reshape to (max_pages * page_size, num_heads, head_dim)
        max_len = max_pages * page_size
        k_flat = k_seq.reshape(max_len, num_heads, head_dim)
        v_flat = v_seq.reshape(max_len, num_heads, head_dim)

        # Compute attention scores: (num_heads, 1, max_len)
        scores = jnp.einsum('qhd,khd->hqk', q_single[0], k_flat) * scale

        # Mask positions beyond actual sequence length
        positions = jnp.arange(max_len)[None, None, :]
        mask = positions < seq_len
        scores = jnp.where(mask, scores, -1e9)

        weights = jax.nn.softmax(scores, axis=-1)
        output = jnp.einsum('hqk,khd->qhd', weights, v_flat)
        return output

    # vmap over batch
    outputs = jax.vmap(single_sequence_attention)(q, page_table, seq_lens)
    return outputs


# ---- 4. Example: Simulate paged KV cache serving ----

print("=== Paged Attention Demo ===\n")

# Configuration
num_pages = 64
page_size = 16
num_heads = 8
head_dim = 64
batch_size = 4
max_pages_per_seq = 8  # supports up to 128 tokens per sequence

# Initialize paged cache
cache = PagedKVCache(num_pages, page_size, num_heads, head_dim)

key = jax.random.PRNGKey(42)

# Simulate 4 sequences with different lengths
actual_seq_lens = jnp.array([45, 80, 30, 100])

# Allocate pages and fill with dummy KV data
page_table = jnp.zeros((batch_size, max_pages_per_seq), dtype=jnp.int32)

for b in range(batch_size):
    num_needed = int(jnp.ceil(actual_seq_lens[b] / page_size))
    for p in range(num_needed):
        phys_page = cache.allocate_page()
        page_table = page_table.at[b, p].set(phys_page)

        # Fill page with random KV data
        key, k1, k2 = jax.random.split(key, 3)
        cache.k_pages = cache.k_pages.at[phys_page].set(
            jax.random.normal(k1, (page_size, num_heads, head_dim))
        )
        cache.v_pages = cache.v_pages.at[phys_page].set(
            jax.random.normal(k2, (page_size, num_heads, head_dim))
        )

print(f"Page table:\n{page_table}")
print(f"Sequence lengths: {actual_seq_lens}")
print(f"Pages allocated: {num_pages - len(cache.free_pages)}/{num_pages}")
print(f"Pages free: {len(cache.free_pages)}/{num_pages}")

# Run paged attention for current query tokens
key, qkey = jax.random.split(key)
q = jax.random.normal(qkey, (batch_size, 1, num_heads, head_dim))

output = paged_attention(
    q, cache.k_pages, cache.v_pages,
    page_table, actual_seq_lens,
    page_size=page_size
)
print(f"\nPaged attention output shape: {output.shape}")


# ---- 5. Memory Comparison ----

max_seq_len = max_pages_per_seq * page_size  # 128
standard_memory = batch_size * max_seq_len * num_heads * head_dim * 2 * 4  # K+V, float32
paged_memory = (num_pages - len(cache.free_pages)) * page_size * num_heads * head_dim * 2 * 4

print(f"\n=== Memory Comparison ===")
print(f"Standard KV cache: {standard_memory / 1024:.1f} KB (pre-allocates max_seq_len)")
print(f"Paged KV cache:    {paged_memory / 1024:.1f} KB (allocates only what's needed)")
print(f"Memory saved:      {(1 - paged_memory / standard_memory) * 100:.1f}%")


# ---- 6. Copy-on-Write for Beam Search ----
# Multiple beams can share prefix pages — only divergent pages are copied

def fork_sequence(page_table, src_seq_idx, dst_seq_idx):
    """
    Fork a sequence for beam search: dst shares src's pages (CoW).
    In a real system, pages are only physically copied when modified.
    """
    return page_table.at[dst_seq_idx].set(page_table[src_seq_idx])

# Beam search example: fork sequence 0 into sequence 3
print(f"\n=== Copy-on-Write Fork ===")
print(f"Before fork - seq 3 pages: {page_table[3]}")
page_table_forked = fork_sequence(page_table, 0, 3)
print(f"After fork  - seq 3 pages: {page_table_forked[3]} (shares seq 0's pages)")

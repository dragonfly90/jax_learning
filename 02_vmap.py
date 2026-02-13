"""
vmap: Vectorized Map (Auto-Batching)
====================================

Key concepts:
- vmap transforms a function that operates on single examples into one
  that operates on batches, without manually writing batch dimensions.
- It's a compiler transformation, not a Python loop — so it's fast.
- in_axes / out_axes control which dimensions to map over.
"""

import jax
import jax.numpy as jnp

# ---- 1. Basic vmap ----
# Suppose we have a function that works on a single vector:

def predict(weights, x):
    """Single prediction: dot product + bias."""
    return jnp.dot(weights[:-1], x) + weights[-1]

key = jax.random.PRNGKey(0)
weights = jax.random.normal(key, (4,))  # 3 weights + 1 bias
single_x = jnp.array([1.0, 2.0, 3.0])

print("Single prediction:", predict(weights, single_x))

# To predict on a BATCH of inputs, use vmap:
batch_x = jax.random.normal(key, (8, 3))  # batch of 8

# Map over axis 0 of x, but broadcast weights (None = don't map)
batch_predict = jax.vmap(predict, in_axes=(None, 0))
predictions = batch_predict(weights, batch_x)
print("Batch predictions shape:", predictions.shape)  # (8,)


# ---- 2. in_axes and out_axes ----
# in_axes specifies which axis of each argument to vectorize over
# None means "broadcast this argument" (don't map over it)

def pairwise_distance(a, b):
    """Euclidean distance between two vectors."""
    return jnp.sqrt(jnp.sum((a - b) ** 2))

points_a = jax.random.normal(key, (5, 3))
points_b = jax.random.normal(key, (5, 3))

# Map over axis 0 of both a and b
batched_dist = jax.vmap(pairwise_distance, in_axes=(0, 0))
distances = batched_dist(points_a, points_b)
print("Pairwise distances:", distances)


# ---- 3. Nested vmap for pairwise computations ----
# Compute ALL pairwise distances (5x5 distance matrix)

# Outer vmap: iterate over rows of points_a
# Inner vmap: for each row of a, iterate over all rows of b
all_pairs_dist = jax.vmap(
    jax.vmap(pairwise_distance, in_axes=(None, 0)),
    in_axes=(0, None)
)
dist_matrix = all_pairs_dist(points_a, points_b)
print("Distance matrix shape:", dist_matrix.shape)  # (5, 5)
print("Distance matrix:\n", dist_matrix)


# ---- 4. vmap + grad: Per-sample gradients ----
# Extremely useful for ML: compute gradients for each sample individually

def single_loss(w, x, y):
    pred = jnp.dot(w, x)
    return (pred - y) ** 2

# grad with respect to w for a single sample
single_grad = jax.grad(single_loss)

# vmap over samples (axis 0 of x and y), broadcast w
per_sample_grads = jax.vmap(single_grad, in_axes=(None, 0, 0))

w = jax.random.normal(key, (3,))
X = jax.random.normal(key, (16, 3))
Y = jax.random.normal(key, (16,))

grads = per_sample_grads(w, X, Y)
print("Per-sample gradients shape:", grads.shape)  # (16, 3)
print("Mean gradient:", jnp.mean(grads, axis=0))


# ---- 5. vmap with jit for performance ----
# Always jit the outer function for best performance

@jax.jit
def fast_batch_predict(weights, batch_x):
    return jax.vmap(predict, in_axes=(None, 0))(weights, batch_x)

result = fast_batch_predict(weights, batch_x)
print("JIT + vmap result shape:", result.shape)


# ---- 6. Practical example: batched matrix-vector multiply ----
# Given a batch of matrices and a batch of vectors, multiply each pair

def matvec(A, x):
    return A @ x

batch_A = jax.random.normal(key, (32, 4, 4))
batch_x = jax.random.normal(key, (32, 4))

batched_matvec = jax.vmap(matvec)
results = batched_matvec(batch_A, batch_x)
print("Batched matvec result shape:", results.shape)  # (32, 4)

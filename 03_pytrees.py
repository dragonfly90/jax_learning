"""
PyTrees: Working with Nested Data Structures
=============================================

Key concepts:
- A pytree is any nested structure of lists, tuples, dicts, and leaves.
- JAX transformations (jit, grad, vmap) work seamlessly with pytrees.
- jax.tree.map applies a function to every leaf in a pytree.
- Pytrees are the standard way to represent model parameters in JAX.
"""

import jax
import jax.numpy as jnp

# ---- 1. What is a PyTree? ----
# Any nested combination of lists, tuples, dicts with array leaves

params = {
    'layer1': {
        'weights': jnp.ones((3, 4)),
        'bias': jnp.zeros(4),
    },
    'layer2': {
        'weights': jnp.ones((4, 2)),
        'bias': jnp.zeros(2),
    },
}

# Count leaves
leaves = jax.tree.leaves(params)
print(f"Number of leaves: {len(leaves)}")
print(f"Leaf shapes: {[l.shape for l in leaves]}")


# ---- 2. jax.tree.map ----
# Apply a function to every leaf

# Double all parameters
doubled = jax.tree.map(lambda x: x * 2, params)
print("Doubled bias:", doubled['layer1']['bias'])

# Element-wise operation on two pytrees with matching structure
params2 = jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, params)
updated = jax.tree.map(lambda p, g: p - 0.01 * g, params, params2)
print("Updated bias:", updated['layer1']['bias'])


# ---- 3. Using pytrees with grad ----
# grad naturally works with pytree-structured parameters

def simple_mlp(params, x):
    """A simple 2-layer MLP."""
    h = jnp.dot(x, params['layer1']['weights']) + params['layer1']['bias']
    h = jax.nn.relu(h)
    out = jnp.dot(h, params['layer2']['weights']) + params['layer2']['bias']
    return out

def loss_fn(params, x, y):
    pred = simple_mlp(params, x)
    return jnp.mean((pred - y) ** 2)

key = jax.random.PRNGKey(42)
x = jax.random.normal(key, (8, 3))
y = jax.random.normal(key, (8, 2))

# grad returns a pytree with the same structure as params
grads = jax.grad(loss_fn)(params, x, y)
print("Gradient keys:", list(grads.keys()))
print("Layer1 weight grad shape:", grads['layer1']['weights'].shape)


# ---- 4. Simple SGD training loop with pytrees ----

@jax.jit
def train_step(params, x, y, lr=0.01):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    # SGD update: params = params - lr * grads
    new_params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    return new_params, loss

for step in range(100):
    params, loss = train_step(params, x, y)
    if step % 20 == 0:
        print(f"Step {step}, Loss: {loss:.4f}")


# ---- 5. Flatten and unflatten ----

flat_params, tree_def = jax.tree.flatten(params)
print(f"\nFlattened: {len(flat_params)} arrays")
print(f"Tree structure: {tree_def}")

# Reconstruct from flat list
reconstructed = tree_def.unflatten(flat_params)
print("Reconstructed keys:", list(reconstructed.keys()))


# ---- 6. Custom pytree nodes ----
# You can register your own classes as pytree nodes

from functools import partial

class MLP:
    def __init__(self, weights, biases):
        self.weights = weights  # list of weight matrices
        self.biases = biases    # list of bias vectors

# Register as a pytree node
def mlp_flatten(mlp):
    children = (mlp.weights, mlp.biases)
    aux_data = None
    return children, aux_data

def mlp_unflatten(aux_data, children):
    return MLP(children[0], children[1])

jax.tree_util.register_pytree_node(MLP, mlp_flatten, mlp_unflatten)

# Now MLP works with all JAX transformations
model = MLP(
    weights=[jnp.ones((3, 4)), jnp.ones((4, 2))],
    biases=[jnp.zeros(4), jnp.zeros(2)]
)
leaves = jax.tree.leaves(model)
print(f"\nCustom MLP leaves: {len(leaves)}")
scaled = jax.tree.map(lambda x: x * 0.5, model)
print("Scaled MLP weight shape:", scaled.weights[0].shape)

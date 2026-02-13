"""
Neural Networks with Flax (Linen API)
======================================

Key concepts:
- Flax is the most popular neural network library for JAX.
- nn.Module: Define layers and models declaratively.
- params = model.init(key, x): Initialize parameters (returns a pytree).
- model.apply(params, x): Forward pass (pure function, no hidden state).
- Optax: Gradient-based optimizers for JAX.

Install: pip install flax optax
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

# ---- 1. Define a simple MLP ----

class MLP(nn.Module):
    """Multi-layer perceptron."""
    hidden_dim: int = 128
    output_dim: int = 10

    @nn.compact
    def __call__(self, x, training: bool = False):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.1)(x, deterministic=not training)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x

# Initialize
model = MLP(hidden_dim=64, output_dim=10)
key = jax.random.PRNGKey(0)
dummy_input = jnp.ones((1, 784))  # e.g., flattened MNIST

# init returns a dict of parameters
variables = model.init(key, dummy_input)
params = variables['params']

print("Parameter tree:")
print(jax.tree.map(lambda x: x.shape, params))


# ---- 2. Forward pass ----

# Inference (no dropout)
logits = model.apply({'params': params}, dummy_input)
print(f"\nOutput shape: {logits.shape}")

# Training (with dropout — need to pass an RNG key)
logits_train = model.apply(
    {'params': params},
    dummy_input,
    training=True,
    rngs={'dropout': jax.random.PRNGKey(1)}
)


# ---- 3. Training loop with Optax ----

# Create optimizer
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)

def cross_entropy_loss(params, x, y, key):
    logits = model.apply(
        {'params': params}, x,
        training=True,
        rngs={'dropout': key}
    )
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))

@jax.jit
def train_step(params, opt_state, x, y, key):
    loss, grads = jax.value_and_grad(cross_entropy_loss)(params, x, y, key)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Simulate training
key = jax.random.PRNGKey(42)
for step in range(100):
    key, subkey, data_key = jax.random.split(key, 3)
    # Fake data
    x = jax.random.normal(data_key, (32, 784))
    y = jax.random.randint(subkey, (32,), 0, 10)

    key, dropout_key = jax.random.split(key)
    params, opt_state, loss = train_step(params, opt_state, x, y, dropout_key)

    if step % 20 == 0:
        print(f"Step {step}: loss={loss:.4f}")


# ---- 4. CNN Example ----

class CNN(nn.Module):
    """Simple CNN for image classification."""
    num_classes: int = 10

    @nn.compact
    def __call__(self, x, training: bool = False):
        # x shape: (batch, height, width, channels)
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape(x.shape[0], -1)  # flatten
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_classes)(x)
        return x

cnn = CNN()
dummy_image = jnp.ones((1, 28, 28, 1))
cnn_params = cnn.init(key, dummy_image)['params']
print("\nCNN parameter shapes:")
print(jax.tree.map(lambda x: x.shape, cnn_params))

logits = cnn.apply({'params': cnn_params}, dummy_image)
print(f"CNN output shape: {logits.shape}")


# ---- 5. Batch Normalization (with state) ----

class ResBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x, training: bool = False):
        residual = x
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Dense(self.features)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        x = nn.Dense(self.features)(x)
        if residual.shape != x.shape:
            residual = nn.Dense(self.features)(residual)
        return x + residual

block = ResBlock(features=64)
variables = block.init(key, jnp.ones((4, 64)), training=True)
print("\nResBlock variable collections:", list(variables.keys()))
# 'params' for weights, 'batch_stats' for running mean/var

# Forward with mutable batch stats
output, updates = block.apply(
    variables,
    jnp.ones((4, 64)),
    training=True,
    mutable=['batch_stats']
)
print(f"ResBlock output shape: {output.shape}")

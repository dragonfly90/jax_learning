# JAX Learning

Hands-on examples covering core JAX concepts, from basics to distributed training.

## Contents

| File | Topic | Key APIs |
|------|-------|----------|
| `01_jax_basics.py` | Arrays, JIT, autodiff | `jnp`, `jit`, `grad`, `value_and_grad`, `jacobian`, `hessian` |
| `02_vmap.py` | Vectorized mapping (auto-batching) | `vmap`, `in_axes`, `out_axes` |
| `03_pytrees.py` | Nested data structures | `tree.map`, `tree.leaves`, `tree.flatten`, custom nodes |
| `04_pmap.py` | Parallel map across devices | `pmap`, `lax.psum`, `lax.pmean` |
| `05_sharding_and_mesh.py` | Modern device parallelism | `Mesh`, `PartitionSpec`, `NamedSharding` |
| `06_scan_and_control_flow.py` | Loops and conditionals in JIT | `lax.scan`, `lax.cond`, `lax.while_loop`, `lax.fori_loop` |
| `07_custom_derivatives.py` | Custom autodiff rules | `custom_jvp`, `custom_vjp`, `stop_gradient`, `checkpoint` |
| `08_nn_with_flax.py` | Neural networks with Flax + Optax | `nn.Module`, `nn.Dense`, `nn.Conv`, `optax.adam` |

## Quick Start

```bash
pip install jax jaxlib flax optax

# Run any example
python 01_jax_basics.py

# For multi-device examples (simulate 8 devices on CPU)
XLA_FLAGS='--xla_force_host_platform_device_count=8' python 05_sharding_and_mesh.py
```

## Key Concepts at a Glance

**Functional paradigm** — JAX functions are pure: no side effects, no hidden state. Parameters are explicit pytrees passed in and out.

**Transformations** — The core of JAX. Each is a function-to-function transform:
- `jit` — compile with XLA for speed
- `grad` — automatic differentiation
- `vmap` — auto-batching
- `pmap` — data parallelism across devices

**Mesh & Sharding** — The modern way to distribute computation. Define a logical device grid (`Mesh`), specify how arrays map to it (`PartitionSpec`), and let `jit` handle the rest.

**Scan** — JAX's loop primitive. Compiles an entire sequential computation (RNN, training loop) into a single efficient XLA program instead of unrolling Python loops.

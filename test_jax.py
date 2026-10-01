import time
import jax
import jaxlib
import jax.numpy as jnp
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
print("JAX:", jax.__version__)
print("jaxlib:", jaxlib.__version__)
print("Devices:", jax.devices())
print("Local devices:", jax.local_devices())

x = jnp.ones((4096, 4096), dtype=jnp.float32)

f = jax.jit(lambda x: x @ x)

print("Starting compilation...", flush=True)
t = time.time()

y = f(x)
y.block_until_ready()

print(f"Done in {time.time()-t:.3f} s", flush=True)
print("Result:", y[0, 0])

import jax
import jax.numpy as jnp
import numpy as np 
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

print("JAX:", jax.__version__)
print("Devices:", jax.devices())

@jax.pmap
def f(x):
    return jax.lax.psum(x, axis_name="i")

# Necesitamos especificar axis_name
f = jax.pmap(
    lambda x: jax.lax.psum(x, "i"),
    axis_name="i",
)

x = jnp.ones((jax.local_device_count(), 1), dtype=jnp.float32)

print("Input:", np.asarray(x), flush=True)

print("Running psum...", flush=True)

y = f(x)

print("pmap returned", flush=True)

y.block_until_ready()

print("block_until_ready OK", flush=True)

for i, shard in enumerate(y.addressable_shards):
    print(
        i,
        shard.device,
        np.asarray(shard.data),
        flush=True,
    )

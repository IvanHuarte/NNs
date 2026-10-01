import time
import numpy as np
import jax
import jax.numpy as jnp
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

print("JAX:", jax.__version__)
print("Devices:", jax.devices())
print("Local devices:", jax.local_devices())

@jax.pmap
def f(x):
    return x * 2.0

x = jnp.ones((jax.local_device_count(), 1000000), dtype=jnp.float32)

print("Starting pmap compilation...", flush=True)
t = time.time()

y = f(x)

print("pmap returned", flush=True)

y.block_until_ready()

print(f"block_until_ready done in {time.time()-t:.3f} s", flush=True)

print("Sharding:", y.sharding, flush=True)
print("Number of addressable shards:", len(y.addressable_shards), flush=True)

for i, shard in enumerate(y.addressable_shards):
    print(f"Shard {i}:", flush=True)
    print("  device:", shard.device, flush=True)
    print("  index:", shard.index, flush=True)
    print("  data type:", type(shard.data), flush=True)

    print("  blocking shard...", flush=True)
    shard.data.block_until_ready()

    print("  copying shard to host...", flush=True)
    z = np.asarray(shard.data)

    print("  copy OK:", z.shape, z[0], flush=True)
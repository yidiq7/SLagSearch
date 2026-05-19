"""Tiny data-parallel sharding helpers shared by gradient_descent.py and plots.py.

Pattern: an outer "device" axis of size D = jax.local_device_count(). Functions
sharded across this axis consume arrays whose leading dim is D, with each
D-slice resident on a distinct local device. The `pmap` is the consumer; this
module only handles packing/unpacking the leading device axis.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def device_put_sharded(shards, devices):
    """Drop-in replacement for the deprecated jax.device_put_sharded.

    Stacks `shards` (a list of D pytrees) into one pytree whose leaves have an
    extra leading axis of size D, with that axis sharded across `devices`.
    """
    mesh = Mesh(np.array(devices), ('x',))
    sharding = NamedSharding(mesh, P('x'))
    return jax.tree.map(
        lambda *xs: jax.device_put(jnp.stack(xs), sharding), *shards
    )


def shard_leading_axis(array, num_devices, devices=None):
    """Pack an (N, ...) array into a (D, N//D, ...) device-sharded array."""
    if devices is None:
        devices = jax.local_devices()
    n = array.shape[0]
    if n % num_devices != 0:
        raise ValueError(
            f"shard_leading_axis: leading dim {n} not divisible by "
            f"num_devices={num_devices}"
        )
    per_device = n // num_devices
    shards = [array[d * per_device:(d + 1) * per_device] for d in range(num_devices)]
    return device_put_sharded(shards, devices)


def unshard_leading_axis(array):
    """Inverse of shard_leading_axis: collapse (D, M, ...) to (D*M, ...)."""
    return array.reshape((-1,) + tuple(array.shape[2:]))


def take_replicated(array):
    """Pmap'd outputs that were pmean'd are replicated across the device axis;
    every slice along axis 0 is identical. Return the first slice as a host-side
    array (still a jnp array, but no device axis)."""
    return array[0]

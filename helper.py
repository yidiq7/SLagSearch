import jax.numpy as jnp

def cluster_phases(phases, tolerance=1e-2):
    if len(phases) == 0:
        return jnp.array([]), jnp.array([])
 
    # Step 1: Wrap to [0, 2π)
    phases_mod = jnp.mod(phases, 2 * jnp.pi)

    # Step 2: Sort and keep track of original indices
    sort_idx = jnp.argsort(phases_mod)
    sorted_phases = phases_mod[sort_idx]

    # Step 3: Compute differences between neighbors (including wrap-around)
    if len(sorted_phases) == 1:
        cluster_ids_sorted = jnp.array([0])
    else:
        # Compute all consecutive differences and the wrap-around difference
        diffs = jnp.concatenate([
            jnp.diff(sorted_phases),
            jnp.array([(sorted_phases[0] + 2 * jnp.pi) - sorted_phases[-1]])
        ])
  
        # Step 4: Identify cluster boundaries
        breaks = diffs > tolerance
 
        # Step 5: Assign cluster IDs (starting from 0)
        cluster_ids_sorted = jnp.concatenate([
            jnp.array([0]), 
            jnp.cumsum(breaks[:-1])
        ])

    # Map back to original order
    cluster_ids = jnp.empty_like(cluster_ids_sorted)
    cluster_ids = cluster_ids.at[sort_idx].set(cluster_ids_sorted)

    return phases_mod, cluster_ids

import jax
import jax.numpy as jnp


@jax.jit
def compute_cosine_similarity(
    states1: jnp.ndarray, states2: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute the cosine similarity between corresponding pairs of states in two arrays.

    Parameters:
    - states1: JAX array of shape (N, D), where N is the number of states
        and D is the dimension.
    - states2: JAX array of shape (N, D), must have the same shape as states1.

    Returns:
    - similarities: JAX array of shape (N,), containing the cosine similarities.
    """
    # Compute the dot product between corresponding states
    dot_products = jnp.sum(states1 * states2, axis=1)

    # Compute the norms (magnitudes) of each state vector
    norms1 = jnp.linalg.norm(states1, axis=1)
    norms2 = jnp.linalg.norm(states2, axis=1)

    # Compute the product of norms
    norm_products = norms1 * norms2

    # Handle cases where norms are zero to avoid division by zero
    # Use jnp.where to avoid division by zero
    safe_norm_products = jnp.where(norm_products == 0, 1.0, norm_products)

    # Compute cosine similarities
    cosine_similarities = dot_products / safe_norm_products

    # Set similarities to zero where norms were zero
    cosine_similarities = jnp.where(norm_products == 0, 0.0, cosine_similarities)

    # Set negative similarities to zero
    cosine_similarities = jnp.maximum(cosine_similarities, 0.25)

    return cosine_similarities

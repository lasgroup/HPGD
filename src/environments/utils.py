import jax
import distrax
from jax import numpy as jnp
from flax import struct

"""
General utility functions for the environments
"""


def sample_array(
    key: jax.random.PRNGKey,
    array: jnp.ndarray,
    logits: jnp.ndarray,
):
    """Sample the given array. If logits are None, sample uniformly."""
    d = distrax.Categorical(logits=logits)
    idx = d.sample(seed=key)
    return array[idx], idx, d.probs


def default_reward_function(
    state: struct.dataclass, action: jnp.ndarray, params: struct.dataclass
) -> jnp.ndarray:
    return jnp.all(state.pos == state.goal, axis=-1).astype(jnp.float32)


"""
ConfigurableFourRooms utility functions
"""

FOUR_ROOMS_DEFAULT_MAP = """
xxxxxxxxxxxxx
x     x     x
x     x     x
x           x
x     x     x
x     x     x
xx xxxx     x
x     xxx xxx
x     x     x
x     x     x
x           x
x     x     x
xxxxxxxxxxxxx"""

FOUR_ROOMS_DEFAULT_CORRIDOR_COORDS = (
    (3, 6),  # Top
    (6, 2),  # Left
    (7, 9),  # Right
    (10, 6),  # Bottom
)


def string_to_bool_map(str_map: str) -> jax.Array:
    """Convert string map into boolean walking map."""
    bool_map = []
    for row in str_map.split("\n")[1:]:
        bool_map.append([r == " " for r in row])
    return jnp.array(bool_map)


def map_project(
    env_map: jax.Array, pos: jnp.ndarray, new_pos: jnp.ndarray
) -> jnp.ndarray:
    """Check if new_pos in the map, if not, stay at pos"""
    in_map = env_map[new_pos[0], new_pos[1]]
    return jax.lax.select(in_map, new_pos, pos)


def get_coordinates(
        str_map: str,
):
    """Get coordinates from the string map."""
    env_map = string_to_bool_map(str_map)
    coordinates = []
    for i in range(env_map.shape[0]):
        for j in range(env_map.shape[1]):
            if env_map[i, j]:
                coordinates.append((i, j))
    return jnp.array(coordinates)


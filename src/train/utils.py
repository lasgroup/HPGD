import collections
from typing import Iterable

from jax import numpy as jnp


def update_nested_pytree(pytree, update: dict):
    """
    Update a nested pytree with another nested dictionary
    :param pytree:
    :param update:
    :return:
    """
    for k, v in update.items():
        if isinstance(v, collections.abc.Mapping):
            pytree = pytree.replace(**{k: update_nested_pytree(eval(f"pytree.{k}"), v)})
        else:
            pytree = pytree.replace(
                **{k: jnp.array(v) if isinstance(v, Iterable) else v}
            )
    return pytree


def remove_non_list_entries(
    d,
    list_parameters=(
        "asset_range",
        "consumption_preferences",
        "prices",
        "consumption_tax_rate",
        "hidden_layers",
    ),
    omit_parameters=("consumption_preferences",),
):
    """
    Remove entries from a dictionary that are not lists (except for the hidden_layers entry)
    Applies to nested dictionaries like config_DQN
    :param d: Nested dictionary
    :param list_parameters: List of parameters that should be lists
    :return:
    """

    def filter_value(k, v):
        if k in omit_parameters:
            return None
        if isinstance(v, dict):
            recursive_output = remove_non_list_entries(v, list_parameters, omit_parameters)
            if len(recursive_output) > 0:
                return recursive_output
            else:
                return None
        elif isinstance(v, list) and (
            k not in list_parameters
            or (k in list_parameters and isinstance(v[0], list))
        ):
            return v
        else:
            return None

    return {
        k: filter_value(k, v) for k, v in d.items() if filter_value(k, v) is not None
    }

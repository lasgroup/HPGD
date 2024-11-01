import jax
import jax.numpy as jnp
import optax
from typing import (
    Dict,
    Union,
    Sequence,
)

from flax.linen.initializers import constant, orthogonal
import flax.linen as nn
from flax.training.train_state import TrainState
import distrax

from functools import reduce
from operator import mul

from gymnax.environments.environment import Environment, EnvParams
from gymnax.environments import spaces


class DQN_Actor(nn.Module):
    action_dim: Union[int, Sequence[int]]
    activation: str = "relu"
    layer_sizes: Sequence[int] = (32,)

    @nn.compact
    def __call__(self, x: jax.Array) -> Union[jax.Array, Sequence[jax.Array]]:
        if self.activation == "relu":
            activation = nn.relu
        elif self.activation == "leaky_relu":
            activation = nn.leaky_relu
        else:
            activation = nn.tanh
        for layer_size in self.layer_sizes:
            x = nn.Dense(
                layer_size, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
            )(x)
            x = activation(x)
        if isinstance(self.action_dim, int):
            return nn.Dense(
                self.action_dim, kernel_init=orthogonal(0.1), bias_init=constant(0.0)
            )(x)
        else:
            """If the action space is multidimensional, we need to create a separate actor for each dimension."""
            q_values = []
            for dim in self.action_dim:
                q_values.append(
                    nn.Dense(dim, kernel_init=orthogonal(0.1), bias_init=constant(0.0))(
                        x
                    )
                )
            return q_values


def create_train_state(
    key: jax.random.PRNGKey,
    config: Dict,
    env: Environment,
    env_params: EnvParams,
) -> TrainState:
    action_space = env.action_space(env_params)
    if isinstance(action_space, spaces.Discrete):
        network = DQN_Actor(
            action_dim=action_space.n,
            activation=config["activation"],
            layer_sizes=config["hidden_layers"],
        )
    elif isinstance(action_space, spaces.Tuple):
        network = DQN_Actor(
            action_dim=reduce(mul, [space.n for space in action_space.spaces])
            if config["correlated_action_dimensions"]
            else [space.n for space in action_space.spaces],
            activation=config["activation"],
            layer_sizes=config["hidden_layers"],
        )
    else:
        raise ValueError("Action space not supported")

    if config["optimizer"]["type"] == "adam":
        tx = optax.chain(
            optax.clip_by_global_norm(max_norm=config["max_grad_norm"]),
            optax.adam(**config["optimizer"]["params"]),
        )
    else:
        raise ValueError("Optimizer not supported")

    return TrainState.create(
        apply_fn=network.apply,
        params=network.init(key, jnp.zeros(env.observation_space(env_params).shape)),
        tx=tx,
    )


def get_actions_from_q_values(
    key: jax.random.PRNGKey,
    q_values: Union[jax.Array, Sequence[jax.Array]],
    reg_lambda: float,
    action_space_shape: Sequence[int],
):
    if isinstance(q_values, list):

        def sample_action_and_entropy(q, _rng):
            action_dist = distrax.Categorical(logits=q / reg_lambda)
            action = action_dist.sample(seed=_rng)
            return action, action_dist.entropy()

        rngs = list(jax.random.split(key, len(q_values)))
        outputs = jax.tree_map(sample_action_and_entropy, q_values, rngs)
        action = jnp.stack(
            jax.tree_map(
                lambda x: x[0], outputs, is_leaf=lambda x: isinstance(x, tuple)
            ),
            axis=-1,
        )
        entropy = jnp.stack(
            jax.tree_map(
                lambda x: x[1], outputs, is_leaf=lambda x: isinstance(x, tuple)
            ),
            axis=-1,
        )
    else:
        action_dist = distrax.Categorical(logits=q_values / reg_lambda)
        action = action_dist.sample(seed=key)  # Shape: (num_envs,)
        action = jnp.unravel_index(action, action_space_shape)
        action = jnp.stack(action, -1)  # Shape: (num_envs, num_actions)
        entropy = action_dist.entropy()
    return action, entropy

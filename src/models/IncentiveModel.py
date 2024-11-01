import flax.linen as nn
import jax.nn
from flax.training.train_state import TrainState
from typing import Dict

import optax
import jax.numpy as jnp

from src.environments.ConfigurableFourRooms import EnvState


def incentive_transform(
    weights: jnp.ndarray,
    activation_function: str = "sigmoid",
    range: jnp.ndarray = (0.0, 1.0),
    temperature: float = 1.0,
) -> jnp.ndarray:
    """
    Transform the incentive model
    :param weights: Weights of the incentive model
    :param activation_function: Activation function
    :param range: Range of the incentives
    :param temperature: Temperature of the incentives
    :return: Parameters of the incentive model
    """
    if activation_function == "sigmoid":
        arr = jax.nn.sigmoid(temperature*weights)
    elif activation_function == "softmax":
        arr = jax.nn.softmax(weights)[:-1]
    else:
        raise ValueError(f"Unknown activation type: {activation_function}")

    arr = (range[1] - range[0])*arr+range[0]
    return arr


class IncentiveModel(nn.Module):
    """
    Static model that returns a single array of parameters when called
    """

    coordinates: jnp.ndarray
    activation_function: str = "sigmoid"
    range: jnp.ndarray = (0.0, 1.0)
    temperature: float = 1.0

    def setup(self):
        """
        Setup the model
        - Softmax activation has one extra parameter for unused incentives
        :return:
        """
        if self.activation_function == "sigmoid":
            self.w = self.param(
                "weights", nn.initializers.normal(), (self.coordinates.shape[0],)
            )
        elif self.activation_function == "softmax":
            self.w = self.param(
                "weights", nn.initializers.normal(), (self.coordinates.shape[0]+1,)
            )
        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")

    @nn.compact
    def __call__(self, state: EnvState, action: jnp.ndarray) -> jnp.ndarray:
        """
        Return the incentive for the given state and action
        :param state: Current state, Shape: (2,)
        :param action: Action, Shape: ()
        :return: Incentive, Shape: ()
        """
        state_mask = jnp.all(state.pos[None, :] == self.coordinates, axis=1)  # Shape: (n_coords,)
        incentives = incentive_transform(self.w, self.activation_function, self.range, self.temperature)
        return jax.lax.select(
            jnp.all(state.pos == state.goal),
            0.0,
            jnp.sum(jnp.where(state_mask, incentives, 0.0)),
        )


def create_incentive_train_state(
    rng: jax.random.PRNGKey,
    config: Dict,
    model_kwargs: dict = {
        "coordinates": None,
    },
) -> TrainState:
    """
    Create the training state for the incentive model
    :param rng: Random number generator
    :param config: Optimisation configuration
    :param model_kwargs: Arguments for the model
    :return: TrainState
    """
    network = IncentiveModel(**model_kwargs)
    init_env_state = EnvState(pos=jnp.zeros(2), goal=jnp.zeros(2), time=0)
    network_params = network.init(
        rng,
        init_env_state,
        jnp.zeros(()),
    )

    if config["optimiser"] == "adam":
        optimiser = optax.adam
    elif config["optimiser"] == "sgd":
        optimiser = optax.sgd
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")

    lr_schedule_config = config["learning_rate_schedule"]
    if lr_schedule_config["type"] == "exponential_decay":
        lr = optax.exponential_decay(
            init_value=config["learning_rate"],
            transition_steps=config["num_outer_iter"],
            decay_rate=lr_schedule_config["args"]["decay_rate"],
            transition_begin=lr_schedule_config["args"]["transition_begin"],
            staircase=False
        )
    elif lr_schedule_config["type"] == "constant":
        lr = config["learning_rate"]
    else:
        raise ValueError(f"Unknown learning rate schedule: {lr_schedule_config['type']}")

    tx = optax.chain(
        optax.clip_by_global_norm(config["max_grad_norm"]),
        optimiser(learning_rate=lr),
    )
    return TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

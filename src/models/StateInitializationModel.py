import flax.linen as nn
from flax.training.train_state import TrainState
import distrax
from typing import Tuple, Dict
import jax

import optax


class StateInitializationModel(nn.Module):
    """
    State Initialization Model that returns a distribution over the statespace when called
    """
    param_shape: Tuple[int] = (2,)
    distribution: distrax.Distribution = distrax.Categorical

    def setup(self):
        self.w = self.param('weights', nn.initializers.normal(), self.param_shape)

    def __call__(self) -> distrax.Distribution:
        """
        Call the model
        """
        return self.distribution(logits=self.w)


def create_state_initialization_train_state(
    rng: jax.random.PRNGKey,
    config: Dict,
    model_kwargs: dict = {
        "param_shape": (2,),
        "distribution": distrax.Categorical,
    },
) -> TrainState:
    """
    Create the training state for the upper level optimisation
    :param rng: Random number generator
    :param config: Optimisation configuration
    :param model_kwargs: Arguments for the model
    :return: TrainState
    """
    network = StateInitializationModel(**model_kwargs)
    network_params = network.init(rng)
    if config["optimiser"] == "adam":
        optimiser = optax.adam
    elif config["optimiser"] == "sgd":
        optimiser = optax.sgd
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    tx = optax.chain(
        optax.clip_by_global_norm(config["max_grad_norm"]),
        optimiser(learning_rate=config["learning_rate"]),
    )
    return TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )


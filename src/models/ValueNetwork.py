import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from typing import Union, Sequence, Dict, Any
import jax.numpy as jnp
import jax
import optax
from flax.training.train_state import TrainState


class ValueNetwork(nn.Module):
    output_dim: Union[int, Sequence[int]]
    activation: str = "relu"
    layer_sizes: Sequence[int] = (32,)

    @nn.compact
    def __call__(self, x: jax.Array) -> Union[jax.Array, Sequence[jax.Array]]:
        if self.activation == "relu":
            activation = nn.relu
        elif self.activation == "tanh":
            activation = nn.tanh
        else:
            raise ValueError(f"Activation {self.activation} not supported")
        for layer_size in self.layer_sizes:
            x = nn.Dense(
                layer_size, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
            )(x)
            x = activation(x)
        return nn.Dense(
            self.output_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(x)


def create_train_state(
    key: jax.random.PRNGKey,
    input_dim: int,
    output_dim: int,
    layer_size: Sequence[int] = (32,),
    activation: str = "relu",
    optimizer_params: Dict[str, Any] = {
        "max_grad_norm": 1.0,
        "learning_Rate": 1e-3,
    },
) -> TrainState:
    """
    Create the training state for the value network
    :param key:
    :param input_dim:
    :param output_dim:
    :param optimizer_params:
    :return:
    """
    model = ValueNetwork(
        layer_sizes=layer_size,
        activation=activation,
        output_dim=output_dim
    )
    tx = optax.chain(
        optax.clip_by_global_norm(optimizer_params["max_grad_norm"]),
        optax.adam(optimizer_params["learning_rate"]),
    )
    return TrainState.create(
        apply_fn=model.apply,
        params=model.init(key, jnp.zeros(input_dim)),
        tx=tx,
    )

@jax.jit
def mse(
    train_state_params: Dict,
    train_state: TrainState,
    x_batched: jax.Array,
    y_batched: jax.Array,
    l2_reg: float = 0.0,
) -> jax.Array:
    """Masked MSE loss"""
    predictions = train_state.apply_fn(train_state_params, x_batched)  # Shape: (n_steps*num_envs, 3)
    # Create a mask where the target is not NaN
    mask = jnp.logical_not(jnp.isnan(y_batched))

    # Apply the mask to both predictions and targets
    masked_predictions = jnp.where(mask, predictions, 0)
    masked_targets = jnp.where(mask, y_batched, 0)

    # Compute L2 regularization of the weights
    l2 = 0.5 * sum(
        jnp.sum(jnp.square(w))
        for w in jax.tree_leaves(train_state_params["params"])
    )

    # Compute the MSE only for the non-NaN elements
    mse = jnp.sum(jnp.square(masked_predictions - masked_targets)) / jnp.sum(mask) + l2_reg * l2

    return mse
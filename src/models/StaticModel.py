import flax.linen as nn
from typing import Sequence, Optional
import jax.numpy as jnp
import jax
import optax
from flax.training.train_state import TrainState

class StaticModel(nn.Module):
    """
    Static model that returns a single array of parameters when called
    """

    param_shape: Sequence[int] = (2,)
    activation_function: str = "linear"
    scale: tuple = None

    def setup(self):
        self.w = self.param("weights", nn.initializers.normal(), self.param_shape)

    def __call__(self) -> jnp.ndarray:
        """
        Call the model
        """
        if self.activation_function == "sigmoid":
            arr = jax.nn.sigmoid(self.w)
        elif self.activation_function == "softmax":
            arr = jax.nn.softmax(self.w)
        elif self.activation_function == "relu":
            arr = jax.nn.relu(self.w)
        elif self.activation_function == "linear":
            arr = self.w
        else:
            raise ValueError(f"Unknown activation type: {self.activation_function}")

        if self.scale is not None:  # Scale the output
            if self.activation_function in ["sigmoid", "softmax"]:
                arr = self.scale[0] + (self.scale[1] - self.scale[0]) * arr
            else:
                raise ValueError(
                    f"Scaling is only supported for sigmoid and softmax activations"
                )
        return arr


def create_state_model(
    param_shape: tuple,
    activation: str,
    scale: tuple,
    learning_rate: float,
    max_grad_norm: float = 1.0,
    optimizer: str = "sgd",
    key: Optional[jax.random.PRNGKey] = jax.random.PRNGKey(0),
    init_value: Optional[jax.Array] = None,
) -> TrainState:
    model = StaticModel(
        param_shape=param_shape,
        activation_function=activation,
        scale=scale,
    )
    if optimizer == "sgd":
        optimizer = optax.sgd
    elif optimizer == "adam":
        optimizer = optax.adam
    else:
        raise ValueError("Optimizer must be either 'sgd' or 'adam'")
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optimizer(learning_rate=learning_rate),
    )
    if init_value is not None:
        weights = (init_value - scale[0]) / (scale[1] - scale[0])
        if activation == "sigmoid":
            weights = jax.scipy.special.logit(weights)
        else:
            raise ValueError("Only sigmoid activation is supported for init_value")
        weights = {"params": {"weights": weights}}
    else:
        weights = model.init(key)

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=weights,
        tx=tx,
    )
    return train_state

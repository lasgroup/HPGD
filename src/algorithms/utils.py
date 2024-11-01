from typing import NamedTuple, Callable, Tuple, Sequence, Dict, Any

import jax.numpy as jnp
import jax
import gymnax
import flax

from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import LogEnvState


from src.environments.ConfigurableFourRooms import (
    EnvState,
    ConfFourRoomsParams,
    ConfigurableFourRooms,
)

"""
UPPER-LEVEL REWARD FUNCTIONS
"""


def get_upper_level_loss_function(
    loss_type: str,
    env: ConfigurableFourRooms,
    env_params: ConfFourRoomsParams,
    scale: float = 1.0,
) -> Callable[[EnvState, int], jnp.ndarray]:
    """
    Get the upper-level loss function
    :param loss_type: Type of the loss function
    :param env: Environment
    :param env_params: Environment parameters
    :param scale: Scaling factor for the reward
    :return: Upper-level loss function
    """
    if loss_type.startswith("corridor_penalty"):
        if loss_type.endswith("_top"):
            corridor_idx = 0
        elif loss_type.endswith("_left"):
            corridor_idx = 1
        elif loss_type.endswith("_right"):
            corridor_idx = 2
        elif loss_type.endswith("_bottom"):
            corridor_idx = 3
        else:
            raise ValueError(f"Unknown corridor type: {loss_type}. Expected: top, left, right, bottom.")
        corridor = env.corridor_coords[corridor_idx]

        def upper_level_loss(state: EnvState, action: int):
            """
            Encourage the default reward scheme while penalize going through the right corridor
            :param state:
            :param action:
            :return:
            """
            env_reward = env.reward_function(
                state, action, env_params.reward_function_params
            )
            penalty_state = jnp.all(state.pos == corridor).astype(jnp.float32)
            return -env_reward + scale*penalty_state

        return upper_level_loss
    elif loss_type == "negative_reward":

        def upper_level_loss(state: EnvState, action: int):
            env_reward = env.reward_function(
                state, action, env_params.reward_function_params
            )
            return -env_reward

        return upper_level_loss
    elif loss_type == "reward":

        def upper_level_loss(state: EnvState, action: int):
            env_reward = env.reward_function(
                state, action, env_params.reward_function_params
            )
            return env_reward

        return upper_level_loss
    else:
        raise ValueError(f"Unknown loss function type: {loss_type}")


"""
UTILITY FUNCTIONS FOR ROLLOUTS
"""


class Transition(NamedTuple):
    done: bool
    t: int
    state: EnvState
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: float
    info: jnp.ndarray


def make_env_step_fn(
    env: gymnax.environments.environment.Environment,
    num_envs: int,
) -> Callable[
    [Tuple[TrainState, EnvState, jnp.ndarray, jax.random.PRNGKey], Any],
    Tuple[Tuple[TrainState, EnvState, jnp.ndarray, jax.random.PRNGKey], Transition],
]:
    """
    Create the environment step function
    :param env: Environment
    :param env_params: Environment parameters
    :param num_envs: Number of environments
    :return: Function to step the environment returning the next state and the transition
        Input: Tuple of (TrainState, EnvState, last_observation (jnp.ndarray), jax.random.PRNGKey), Any
        Output: Tuple of (TrainState, EnvState, new_observation (jnp.ndarray), jax.random.PRNGKey), Transition
    """

    def _env_step(
        env_step_state: Tuple[TrainState, NamedTuple, EnvState, jnp.ndarray, jax.random.PRNGKey],
        unused: Any,
    ) -> Tuple[Tuple[TrainState, NamedTuple, EnvState, jnp.ndarray, jax.random.PRNGKey], Transition]:
        """
        Carry out one step in the environment
        :param env_step_state: Tuple of (TrainState, EnvState, last_observation (jnp.ndarray), jax.random.PRNGKey)
            - TrainState: TrainState of the agent i.e. Policy and Value function
        :param unused:
        :return:
        """
        train_state, env_params, last_env_state, last_obs, rng = env_step_state

        # SELECT ACTION
        rng, _rng = jax.random.split(rng)
        pi, value = train_state.apply_fn(train_state.params, last_obs)
        action = pi.sample(seed=_rng)

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, num_envs)
        obsv, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(rng_step, last_env_state, action, env_params)

        # ADJUST ENV STATE IF ENV WRAPPER IS USED
        if isinstance(last_env_state, LogEnvState):
            last_env_state = last_env_state.env_state
        t = (
            env_state.env_state.time
            if hasattr(env_state, "env_state")
            else env_state.time
        )
        transition = Transition(
            done,
            t,
            last_env_state,
            last_obs,
            action,
            reward,
            info,
        )
        env_step_state = (train_state, env_params, env_state, obsv, rng)
        return env_step_state, transition

    return _env_step


def normalize_discounted_rewards(
    discounted_rewards: jnp.ndarray,
) -> jnp.ndarray:
    """
    Normalize the discounted rewards
    :param discounted_rewards: Discounted rewards
    :return: Normalized discounted rewards
    """
    mean = jnp.nanmean(discounted_rewards)
    mean2 = jnp.nanmean(discounted_rewards**2)
    std = jnp.sqrt(mean2 - mean**2) + 1e-16
    discounted_rewards_normalized = (discounted_rewards - mean) / std
    return discounted_rewards_normalized


def calculate_discounted_episode_reward(
    traj_batch: Sequence[Transition],
    reward_transformation: Callable[[Transition], float],
    baseline: Tuple[jnp.ndarray, jnp.ndarray] = None,
    return_initial_only: bool = False,
    normalize: bool = True,
    gamma: float = 1.0,
    num_envs: int = 4,
) -> jnp.ndarray:
    """
    Calculate the normalized discounted episode reward for each episode in the batch.
    :param traj_batch: Batch of trajectories returned by the sampling
    :param reward_transformation: Reward transformation applied on the Transition namedtuple in the trajectories
    :param baseline: Tuple of (agent_positions, baseline_values) for the baseline subtraction
        For example, agent_positions can be the coordinates in a gridworld environment, i.e., of Shape: (num_cells, 2)
        Baseline values are defined accordingly to the agent_positions, Shape: (num_cells, )
    :param return_initial_only: If true, only return the discounted reward for the initial state of each episode
        Intermediate rewards are set to NaN
    :param normalize: If true, normalize the discounted rewards
    :param gamma: Discount factor
    :param num_envs: Number of environments
    :return: Normalized discounted episode rewards
    """

    def _get_discounted_reward(
        rolling_discounted_rewards: jnp.ndarray,
        transition: Transition,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        done, reward = transition.done, reward_transformation(transition)
        rolling_discounted_rewards = reward + gamma * rolling_discounted_rewards * (
            1 - done
        )  # Calculate discounted reward, Shape: (num_envs,)
        discounted_rewards_output = jax.lax.select(
            jnp.logical_and(
                jnp.logical_or(
                    transition.t != 1,
                    jnp.abs(rolling_discounted_rewards) < 1e-7,
                ),  # If not initial or reward is close to 0
                jnp.full_like(transition.t, return_initial_only),
            ),
            jnp.full_like(
                rolling_discounted_rewards, jnp.nan
            ),  # On True, if return_initial_only
            rolling_discounted_rewards,  # On False, if return_initial_only is False
        )  # Set intermediate rewards to NaN if return_initial_only
        return rolling_discounted_rewards, discounted_rewards_output

    _, discounted_rewards = jax.lax.scan(
        _get_discounted_reward,
        jnp.zeros(num_envs),
        traj_batch,
        reverse=True,
    )
    discounted_rewards = jax.lax.cond(
        normalize,
        lambda x: normalize_discounted_rewards(x),
        lambda x: x,
        discounted_rewards,
    )
    return discounted_rewards


"""
GRADIENT LOSS FUNCTIONS
"""


def policy_gradient_loss(
    params: Dict,
    batch: Transition,
    train_state: flax.training.train_state.TrainState,
    normalized_discounted_rewards: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate the policy gradient loss for the given batch of transitions."""
    pi, _ = train_state.apply_fn(params, batch.obs)
    log_prob = pi.log_prob(batch.action)
    policy_gradients = -log_prob * normalized_discounted_rewards
    return jnp.mean(policy_gradients)


def initialization_gradient_estimator(
    params: Dict,
    batch: Transition,
    normalized_discounted_rewards: jnp.ndarray,
    env: ConfigurableFourRooms,
) -> jnp.ndarray:
    """
    Calculate the initialization gradient loss for the given batch of transitions.
        score_objective = E_{\tau ~ Pi(\theta; x)} [G_0 \nabla_x log p_0(s_0; x)]
    Input Shape: batch.obs Shape: (num_steps, num_envs, obs_shape)
    """
    is_initial_position = jnp.all(
        batch.state.pos[..., None] == env.available_init_pos.transpose()[None, None, ...],
        axis=-2,
    )  # Shape: (num_steps, num_envs, num_init_pos)
    is_initial_time = jnp.equal(batch.t, 1)  # Shape: (num_steps, num_envs)
    is_initial_position = jnp.logical_and(
        is_initial_position,
        is_initial_time[:, :, None]
    )  # Shape: (num_steps, num_envs, num_init_pos)

    init_dist = env.state_initialization_distribution(params)
    # Shape: (num_init_pos,)
    x = jnp.arange(env.available_init_pos.shape[0])
    log_prob = init_dist.log_prob(x)  # Shape: (num_init_pos,)
    loss = (
        is_initial_position
        * log_prob[None, None, :]
        * jnp.expand_dims(normalized_discounted_rewards, axis=-1)
    )  # Shape: (num_steps, num_envs, num_init_pos)
    # Numerical issues with jnp.nanmean in gradient calculations, Use jnp.nansum instead
    num_entries = jnp.sum(is_initial_position)  # Shape: (num_envs, num_init_pos)
    return jnp.nansum(loss)/num_entries


# Vectorizing over the batch dimension
initialization_gradient_estimation_vmap = jax.vmap(
    initialization_gradient_estimator,
    in_axes=(None, 0, 0, None),
)


def incentive_gradient_estimator(
    params: Dict,
    batch: Sequence[Transition],
    env: ConfigurableFourRooms,
    gamma: float = 1.0,
) -> jnp.ndarray:
    """
    Calculate the initialization gradient loss for the given batch of transitions.
        score_objective = E_{\tau ~ Pi(\theta; x)} [\nabla G_0(tau; x)]
    Input Shape: batch.obs Shape: (num_steps, num_envs, obs_shape)
    """
    incentive_f_vmap = jax.vmap(
        env.incentive_function,
        in_axes=(0, 0, None),
    )  # vmap over num_envs

    def get_instant_incentive(
            rolling_discounted_incentive,
            transition,
    ):
        incentive = incentive_f_vmap(
            transition.state,
            transition.action,
            params,
        )  # Shape: (num_envs,)
        rolling_discounted_incentive = jnp.where(
            transition.done,
            incentive,
            incentive + gamma * rolling_discounted_incentive
        )
        return rolling_discounted_incentive, rolling_discounted_incentive

    _, discounted_incentives = jax.lax.scan(
        get_instant_incentive,
        jnp.zeros(batch.action.shape[1]),
        batch,
        reverse=True
    )
    return jnp.mean(discounted_incentives)


# Vectorizing over the batch dimension
incentive_gradient_estimator_vmap = jax.vmap(
    incentive_gradient_estimator,
    in_axes=(None, 0, None, None),
)


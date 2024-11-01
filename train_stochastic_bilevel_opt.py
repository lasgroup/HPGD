import argparse
import os

import flax.core.frozen_dict
import yaml
import pickle
import time
from typing import Dict, Tuple, Callable, Iterable

import jax
import jax.numpy as jnp
import distrax
import orbax
import optax
from gymnax.wrappers.purerl import FlattenObservationWrapper
from flax.training import orbax_utils
from flax.training.train_state import TrainState

from src.environments.ConfigurableFourRooms import ConfigurableFourRooms
from src.environments.utils import (
    FOUR_ROOMS_DEFAULT_CORRIDOR_COORDS,
    FOUR_ROOMS_DEFAULT_MAP,
    get_coordinates,
    sample_array,
)

from src.models.IncentiveModel import create_incentive_train_state, incentive_transform

from src.algorithms.value_iteration_and_prediction import (
    general_value_iteration,
    initial_value_prediction,
)
from src.algorithms.utils import Transition, make_env_step_fn


def clip_norm(arr, max_norm):
    """
    Clip the gradient norm
    :param grad:
    :param max_norm:
    :return:
    """
    norm = jnp.linalg.norm(
        jnp.array(jax.tree_util.tree_leaves(jax.tree_map(jnp.linalg.norm, arr)))
    )
    clip = lambda x: jnp.where(norm < max_norm, x, x * max_norm / (norm + 1e-6))
    return jax.tree_util.tree_map(clip, arr)


def regularized_softmax(Q: jnp.ndarray, reg_lambda: jnp.ndarray) -> jnp.ndarray:
    """
    Regularized softmax function
    :param Q: Q-value table, Shape: (n_states, n_actions)
    :param reg_lambda: Regularization parameter, Shape: ()
    :return:
    """
    return jax.nn.softmax(Q / reg_lambda, axis=-1)


def apply_greedy_policy(
    train_state_params: Dict[str, jnp.ndarray],
    obs: jnp.ndarray,
    state_coords: jnp.ndarray,
) -> Tuple[distrax.Distribution, jnp.ndarray]:
    """
    Apply the best response policy to the environment
    :param train_state_params: Parameters of the TrainState variable
    :param obs: Observations, Shape: (n_envs, |obs|)
    :param state_coords: Coordinates of the state, Shape: (n_coords, 2)
    :return:
    """
    pos = obs[:, :2]  # (n_envs, 2)
    mask_position = jnp.all(
        state_coords[:, :, None] == pos.T[None, ...], 1
    )  # (n_coords, n_envs)
    probs = jnp.dot(
        mask_position.T, train_state_params["best_response_policy"]
    )  # (n_envs, n_actions)
    dist = distrax.Categorical(probs=probs)
    return dist, jnp.nan


def upper_level_reward(
    params: Dict[str, jnp.ndarray],
    state: jnp.ndarray,
    action: jnp.ndarray,
    config: Dict,
):
    """
    Penalize going through the bottom door
    Input shapes:
    - state.pos: (2,)
    - state.goal: (2,)
    """
    config_upper = config["upper_optimisation"]
    if config_upper["reward_function"]["type"] == "positive":
        sign = 1
    elif config_upper["reward_function"]["type"] == "negative":
        sign = -1
    else:
        raise ValueError("Invalid reward function type")
    raw_reward = sign * jnp.all(
        state.pos == jnp.array(config_upper["reward_function"]["target_state"])
    ).astype(jnp.float32)

    config_incentive = config["configuration"]["incentive"]
    is_goal = jnp.all(state.goal == state.pos).astype(jnp.float32)
    incentives = incentive_transform(
        params["params"]["weights"],
        activation_function=config_incentive["activation_function"],
        range=config_incentive["range"],
        temperature=config_incentive["temperature"],
    )
    regularization = -is_goal * jnp.linalg.norm(incentives, ord=1)
    return (
        raw_reward
        + config_upper["incentive_reg_param"] * regularization
    )


def environment_setup(rng: jax.random.PRNGKey, config: Dict) -> Tuple:
    """
    Setup the environment
    :param rng:
    :param config:
    :return:
    """
    rng, _rng_incentive = jax.random.split(rng, 2)
    config_incentive = config["configuration"]["incentive"].copy()
    if config_incentive["coordinates"] == "all":
        incentive_coords = get_coordinates(FOUR_ROOMS_DEFAULT_MAP)
    elif config_incentive["coordinates"] == "corridors":
        incentive_coords = jnp.array(FOUR_ROOMS_DEFAULT_CORRIDOR_COORDS)
    else:
        raise ValueError("Invalid incentive coordinates")
    config_incentive["coordinates"] = incentive_coords
    incentive_train_state = create_incentive_train_state(
        _rng_incentive, config["upper_optimisation"], model_kwargs=config_incentive
    )
    config_env = config["environment"]
    env = ConfigurableFourRooms(
        available_goals=config_env["available_goals"],
        incentive_function=lambda state, action, params: incentive_train_state.apply_fn(
            params, state, action
        ),
    )
    env_params = env.default_params
    init_coords = jnp.array(config_env["available_init_pos"])  # (n_init_pos, 2)
    resample_init_pos_prob = jnp.array(
        config_env["resample_init_pos_prob"]
    )  # (n_init_pos,)
    initialization_probs = jnp.all(
        env.coords[..., None] == init_coords.T[None, ...], axis=1
    )  # (n_states, n_init_pos)
    if jnp.nansum(resample_init_pos_prob) > 0:
        initialization_probs = jnp.sum(
            initialization_probs * resample_init_pos_prob[None, :], axis=1
        )
    else:
        initialization_probs = jnp.sum(initialization_probs, axis=1)
    env_params = env_params.replace(
        fail_prob=config_env["fail_prob"],
        max_steps_in_episode=config_env["max_steps_in_episode"],
        state_initialization_params=jnp.log(
            initialization_probs + 1e-16
        ),  # Logit transformation
        incentive_params=incentive_train_state.params,
    )
    return env, env_params, incentive_train_state, config_incentive


def calculate_discounted_rewards(
    reward_function_params,
    reward_function,
    traj_batch: Transition,
    discount_factor: float,
    n_envs: int,
) -> jnp.ndarray:
    """
    Calculate the discounted rewards for a trajectory batch
    :param reward_function_params:
    :param reward_function:
    :param traj_batch:
    :param discount_factor:
    :param n_envs:
    :return: Discounted rewards, Shape: (n_steps, num_envs)
    """

    def _get_discounted_reward(
        rolling_discounted_rewards: jnp.ndarray,
        transition: Transition,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # vmap over num_envs dimension
        reward = jax.vmap(
            reward_function,
            in_axes=(None, 0, 0),
        )(
            reward_function_params, transition.state, transition.action
        )  # Shape: (num_envs, output_dim(optional) )
        done = transition.done.astype(jnp.float32)  # Shape: (num_envs,)
        rolling_discounted_rewards = (
            reward + discount_factor * rolling_discounted_rewards * (1 - done)
        )  # Calculate discounted reward, Shape: (num_envs,)
        return rolling_discounted_rewards, rolling_discounted_rewards

    _, discounted_rewards = jax.lax.scan(
        _get_discounted_reward,
        jnp.zeros(n_envs),
        traj_batch,
        reverse=True,
    )  # Shape: (n_steps, num_envs)
    return discounted_rewards


def estimate_value_function(
    traj_batch: Transition,
    discounted_rewards: jnp.ndarray,
    state_coords: jnp.ndarray,
    n_actions: int,
    value_function="value",
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Estimates the value and Q-function for the given trajectory batch by averaging the discounted rewards
    """
    assert value_function in ["value", "Q"], "Invalid value function type"

    # Flatten the trajectory batch dimensions for easier processing
    flat_pos = traj_batch.state.pos.reshape(-1, 2)  # (n_steps*n_envs, 2)
    if len(discounted_rewards.shape) == 2:
        flat_rewards = discounted_rewards.ravel()  # (n_steps*n_envs,)
    elif len(discounted_rewards.shape) == 3:
        flat_rewards = discounted_rewards.reshape(
            -1, discounted_rewards.shape[-1]
        )  # (n_steps*n_envs, |params|)
    else:
        raise ValueError("Invalid discounted rewards shape")

    # Find state indices by matching positions to state coordinates
    # This generates a boolean mask with shape (n_steps*n_envs, n_states) indicating matches
    matches = jnp.all(
        flat_pos[:, None, :] == state_coords[None, :, :], axis=-1
    )  # (n_steps*n_envs, n_states)

    # Assume exactly one match per position for simplicity; obtain index of matching state
    state_indices = jnp.argmax(matches, axis=-1)  # (n_steps*n_envs,)

    def compute_values_for_states(values, indices, num_indices):
        sums = jax.ops.segment_sum(values, indices, num_segments=num_indices)
        counts = jax.ops.segment_sum(
            jnp.ones_like(values), indices, num_segments=num_indices
        )
        return sums / jnp.maximum(counts, 1)  # Avoid division by zero

    # Computing value function or Q-function as requested
    if value_function == "value":
        n_states = state_coords.shape[0]
        flat_actions = None
        avg_discounted_rewards = compute_values_for_states(
            flat_rewards, state_indices, n_states
        )
    else:  # value_function == "Q"
        # Create composite state-action indices for segment operations
        flat_actions = traj_batch.action.ravel()  # (n_steps*n_envs,)
        composite_indices = state_indices * n_actions + flat_actions
        n_composite = state_coords.shape[0] * n_actions
        avg_discounted_rewards = compute_values_for_states(
            flat_rewards, composite_indices, n_composite
        )  # Shape: (n_states*n_actions, |params|)
        # Reshape back to separate states and actions
        if len(discounted_rewards.shape) == 2:
            avg_discounted_rewards = avg_discounted_rewards.reshape(
                -1, n_actions
            )  # (n_states, n_actions)
        else:
            avg_discounted_rewards = avg_discounted_rewards.reshape(
                -1, n_actions, discounted_rewards.shape[-1]
            )  # (n_states, n_actions, |params|)
    return avg_discounted_rewards, state_indices, flat_actions


def create_update_step(
    env: ConfigurableFourRooms,
    config: Dict,
) -> Callable:
    """
    Create the update step function for the bilevel optimization
    """
    config_lower_level = config["lower_optimisation"]
    config_upper_level = config["upper_optimisation"]
    config_upper_level["num_steps_per_update"] = (
        config_upper_level["num_total_steps"] // config_upper_level["num_envs"]
    )

    env = FlattenObservationWrapper(env)
    _env_step = make_env_step_fn(
        env=env,
        num_envs=config_upper_level["num_envs"],
    )

    def update_step(carry, reg_lambda: float) -> Tuple:
        (rng, env_params_train_carry, incentive_train_state_carry) = carry

        # Realize stochasticity in the environment
        rng, _rng = jax.random.split(rng)
        goal, xi_idx, goal_probs = sample_array(
            _rng, env.available_goals, env_params_train_carry.resample_goal_logits
        )
        env_params_fixed_xi = env_params_train_carry.replace(
            resample_goal_logits=jnp.log(
                jnp.full_like(env_params_train_carry.resample_goal_logits, 1e-16)
                .at[xi_idx]
                .set(1.0)
            )
        )

        # Calculate best-response
        q_final, _ = general_value_iteration(
            env,
            env_params_train_carry,
            gamma=config_lower_level["discount_factor"],
            n_policy_iter=config_lower_level["n_policy_iter"],
            n_value_iter=config_lower_level["n_value_iter"],
            regularization=config_lower_level["regularization"],
            reg_lambda=reg_lambda,
            return_q_value=True,
        )
        policy = regularized_softmax(
            q_final[xi_idx], reg_lambda
        )  # Shape: (n_states, n_actions)
        train_state = TrainState.create(
            apply_fn=lambda params, obs: apply_greedy_policy(params, obs, env.coords),
            params={"best_response_policy": policy},
            tx=optax.identity(),
        )

        # ROLLOUT, TRAJECTORY COLLECTION
        (
            rng,
            _rng_reset,
            _rng_rollout,
            _rng_reset_uniform,
            _rng_rollout_uniform,
        ) = jax.random.split(rng, 5)
        _rng_reset = jax.random.split(_rng_reset, config_upper_level["num_envs"])
        _rng_reset_uniform = jax.random.split(
            _rng_reset_uniform, config_upper_level["num_envs"]
        )
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(
            _rng_reset,
            env_params_fixed_xi,
        )
        env_step_carry_state = (
            train_state,
            env_params_fixed_xi,
            env_state,
            obsv,
            _rng_rollout,
        )
        _, traj_batch = jax.lax.scan(
            _env_step,
            env_step_carry_state,
            None,
            config_upper_level["num_steps_per_update"],
        )  # Shape: (num_steps_per_update, num_envs, n_obs/n_state/None)

        def get_uniform_initialized_traj_batch() -> Transition:
            """
            Get the trajectory batch with uniform initialization
            """
            _rng_rollout_uniform_split, _rng_init = jax.random.split(
                _rng_rollout_uniform
            )
            random_idx = jax.random.choice(
                _rng_init,
                env_params_fixed_xi.state_initialization_params.shape[0],
            )
            logit_single_random = jnp.log(
                jnp.full_like(env_params_fixed_xi.state_initialization_params, 1e-16)
                .at[random_idx]
                .set(1.0)
            )
            initialization_logits = jax.lax.select(
                config_upper_level["advantage_gradient_sampling"] == "uniform",
                jnp.ones_like(
                    env_params_fixed_xi.state_initialization_params
                ),  # Uniform sampling
                logit_single_random,  # Single random initialization
            )
            env_params_fixed_xi_uniform_init = env_params_fixed_xi.replace(
                state_initialization_params=initialization_logits
            )
            obsv_uniform, env_state_uniform = jax.vmap(env.reset, in_axes=(0, None))(
                _rng_reset_uniform,
                env_params_fixed_xi_uniform_init,
            )
            env_step_carry_state_uniform = (
                train_state,
                env_params_fixed_xi_uniform_init,
                env_state_uniform,
                obsv_uniform,
                _rng_rollout_uniform_split,
            )
            _, traj_batch_uniform = jax.lax.scan(
                _env_step,
                env_step_carry_state_uniform,
                None,
                config_upper_level["num_steps_per_update"],
            )  # Shape: (num_steps_per_update, num_envs, n_obs/n_state/None)
            return traj_batch_uniform

        # If the advantage gradient sampling is uniform, initialize the trajectory batch with uniform initialization and sample a new batch
        traj_batch_advantage = jax.lax.cond(
            config_upper_level["advantage_gradient_sampling"]
            in ["uniform", "uniform_single"],
            get_uniform_initialized_traj_batch,
            lambda: traj_batch,
        )

        # GRADIENT CALCULATION
        # LL ADVANTAGE
        discounted_rewards_grad = jax.jacfwd(
            calculate_discounted_rewards,
        )(
            env_params_fixed_xi.incentive_params,
            lambda param, s, a: env.incentive_function(s, a, param),
            traj_batch_advantage,
            config_lower_level["discount_factor"],
            config_upper_level["num_envs"],
        )  # (n_steps, n_envs, |params|)
        discounted_rewards_grad = discounted_rewards_grad["params"][
            "weights"
        ]  # (n_steps, n_envs, |params|)
        LL_value_grad, _, _ = estimate_value_function(
            traj_batch_advantage,
            discounted_rewards_grad,
            env.coords,
            env.action_space().n,
            "value",
        )  # (n_coords, |params|)
        LL_Q_grad, state_indices, action_indices = estimate_value_function(
            traj_batch_advantage,
            discounted_rewards_grad,
            env.coords,
            env.action_space().n,
            "Q",
        )  # (n_coords, n_actions, |params|)
        LL_advantage_grad = LL_Q_grad - jnp.expand_dims(
            LL_value_grad, 1
        )  # (n_coords, n_actions, |params|)

        # Mask gradient back to traj_batch_advantage dimensions
        LL_advantage_grad = LL_advantage_grad[state_indices, action_indices, :].reshape(
            config_upper_level["num_steps_per_update"],
            config_upper_level["num_envs"],
            -1,
        )  # (n_steps, n_envs, |params|)

        # Calculate upper-level discounted rewards ~ Q(s, a)
        UL_discounted_rewards = calculate_discounted_rewards(
            env_params_fixed_xi.incentive_params,
            lambda p, s, a: upper_level_reward(p, s, a, config),
            traj_batch_advantage,
            config_upper_level["discount_factor"],
            config_upper_level["num_envs"],
        )  # (n_steps, n_envs)
        UL_value, _, _ = estimate_value_function(
            traj_batch_advantage,
            UL_discounted_rewards,
            env.coords,
            env.action_space().n,
            "value",
        )  # (n_states, )
        UL_Q_value, state_indices, action_indices = estimate_value_function(
            traj_batch_advantage,
            UL_discounted_rewards,
            env.coords,
            env.action_space().n,
            "Q",
        )
        UL_advantage = UL_Q_value - jnp.expand_dims(
            UL_value, 1
        )  # (n_states, n_actions)
        UL_advantage = UL_advantage[state_indices, action_indices].reshape(
            config_upper_level["num_steps_per_update"],
            config_upper_level["num_envs"],
        )
        br_response_grad = (
            LL_advantage_grad * jnp.expand_dims(UL_advantage, -1) / reg_lambda
        )  # (n_steps, n_envs, |params|)
        UL_discounted_rewards_grad = jax.jacfwd(
            calculate_discounted_rewards,
        )(
            env_params_train_carry.incentive_params,
            lambda p, s, a: upper_level_reward(p, s, a, config),
            traj_batch,
            config_upper_level["discount_factor"],
            config_upper_level["num_envs"],
        )  # (n_steps, n_envs, |params|)
        UL_discounted_rewards_grad = UL_discounted_rewards_grad["params"][
            "weights"
        ]  # (n_steps, n_envs, |params|)
        UL_value_grad, state_indices, _ = estimate_value_function(
            traj_batch,
            UL_discounted_rewards_grad,
            env.coords,
            env.action_space().n,
            "value",
        )  # (n_coords, |params|)
        UL_value_grad = UL_value_grad[state_indices, :].reshape(
            config_upper_level["num_steps_per_update"],
            config_upper_level["num_envs"],
            -1,
        )  # (n_steps, n_envs, |params|)

        # Shift the time grid, originally when traj_batch.done then traj_batch.t == 0
        time_grid = jnp.where(
            traj_batch_advantage.done[1:],
            traj_batch_advantage.t[:-1],
            traj_batch_advantage.t[1:] - 1,
        )
        time_grid = jnp.concatenate(
            [jnp.zeros((1, config_upper_level["num_envs"])), time_grid], axis=0
        )
        traj_batch_advantage_discount_factor = jnp.power(
            config_upper_level["discount_factor"],
            time_grid,  # Adjust counting since it starts from 1
        )[
            :, :, None
        ]  # (n_steps, n_envs, 1)
        n_episodes_advantage = jnp.sum(traj_batch_advantage.done)
        br_response_grad = (
            jnp.sum(
                br_response_grad * traj_batch_advantage_discount_factor, axis=(0, 1)
            )
            / n_episodes_advantage
        )  # Average over num episode finished
        UL_reward_grad = jnp.nanmean(
            jnp.where(jnp.expand_dims(traj_batch.t == 1, 2), UL_value_grad, jnp.nan),
            axis=(0, 1),
        )
        grad = UL_reward_grad + br_response_grad  # (|params|)

        # Update the parameters
        # Overall negative for gradient ascent, negate reward_loss_grad due to lower-level maximization
        update_grad = {"params": {"weights": -grad}}
        incentive_train_state_carry = incentive_train_state_carry.apply_gradients(
            grads=flax.core.frozen_dict.FrozenDict(update_grad)
            if jax.__version__ == "0.4.10"
            else update_grad
        )
        env_params_train_carry = env_params_train_carry.replace(
            incentive_params=incentive_train_state_carry.params
        )
        # Statistics to report
        UL_discounted_rewards = calculate_discounted_rewards(
            env_params_fixed_xi.incentive_params,
            lambda p, s, a: upper_level_reward(p, s, a, config),
            traj_batch,
            config_upper_level["discount_factor"],
            config_upper_level["num_envs"],
        )  # (n_steps, n_envs), Calculate the discounted rewards for the trajectory batch (not advantage_traj_batch)
        UL_initial_value_estimate = jnp.nanmean(
            jnp.where(traj_batch.t == 1, UL_discounted_rewards, jnp.nan)
        )
        V_UL, _ = initial_value_prediction(
            env,
            env_params_fixed_xi,
            gamma=config_upper_level["discount_factor"],
            n_policy_iter=config_lower_level["n_policy_iter"],
            n_value_iter=config_lower_level["n_value_iter"],
            policy=policy,
            external_reward=lambda s, a, env_p: upper_level_reward(
                env_p.incentive_params, s, a, config
            ),
        )  # Shape: ()
        num_episodes = jnp.sum(traj_batch.done)
        average_episode_length = (
            traj_batch.done.shape[0]
            * traj_batch.done.shape[1]
            / jnp.clip(num_episodes, a_min=1)
        )
        return (rng, env_params_train_carry, incentive_train_state_carry), {
            "xi_idx": xi_idx,
            "UL_initial_value_estimate": UL_initial_value_estimate,
            "UL_initial_value": V_UL,
            "num_episodes": num_episodes,
            "average_episode_length": average_episode_length,
            "policy_grad_norm": jnp.linalg.norm(br_response_grad),
            "UL_reward_grad_norm": jnp.linalg.norm(UL_reward_grad),
        }

    return update_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dir", type=str, help="Path to the experiment directory"
    )
    parser.add_argument(
        "--advantage_gradient_sampling",
        type=str,
        help="Advantage gradient sampling",
        default="on_policy",
        choices=["on_policy", "uniform", "uniform_single"],
    )
    args = parser.parse_args()
    experiment_dir = args.experiment_dir
    print("Output directory: ", experiment_dir)
    print("Device used: ", jax.devices())

    config = yaml.safe_load(open(os.path.join(experiment_dir, "config.yaml"), "r"))
    config["upper_optimisation"][
        "advantage_gradient_sampling"
    ] = args.advantage_gradient_sampling
    rng = jax.random.PRNGKey(config["random_seed"])

    # Setup environment
    rng, _rng = jax.random.split(rng)
    env, env_params, incentive_train_state, config_incentive = environment_setup(
        _rng, config
    )

    def run_experiment(
        rng: jnp.ndarray,
        upper_optimisation_lr: float,
        upper_optimisation_incentive_reg: float,
        lower_optimisation_reg_lambda_decay: float,
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, TrainState], Tuple[jnp.ndarray]]:
        config_exp = config.copy()
        config_exp["upper_optimisation"]["learning_rate"] = upper_optimisation_lr
        config_exp["upper_optimisation"][
            "incentive_reg_param"
        ] = upper_optimisation_incentive_reg

        # Incentive model
        rng, _rng_incentive = jax.random.split(rng)
        incentive_train_state_exp = create_incentive_train_state(
            _rng_incentive,
            config_exp["upper_optimisation"],
            model_kwargs=config_incentive,
        )
        env_params_exp = env_params.replace(
            incentive_params=incentive_train_state_exp.params,
        )

        # TRAINING
        update_step = create_update_step(env, config_exp)
        n_iter = config_exp["upper_optimisation"]["num_outer_iter"]
        reg_lambda = config["lower_optimisation"]["reg_lambda"] * jnp.power(lower_optimisation_reg_lambda_decay, jnp.arange(n_iter))
        return jax.lax.scan(
            update_step,
            (rng, env_params_exp, incentive_train_state_exp),
            reg_lambda,
            n_iter,
        )

    start_time = time.time()
    config_upper = config["upper_optimisation"]
    # Run grid search if learning rate or incentive reg param is iterable
    if (
        isinstance(config_upper["learning_rate"], Iterable)
        or isinstance(config_upper["incentive_reg_param"], Iterable)
        or isinstance(config["lower_optimisation"]["reg_lambda_decay"], Iterable)
    ):
        print("Running grid search")
        lr_grid, incentive_reg_grid, lambda_decay_grid = jnp.meshgrid(
            jnp.atleast_1d(config_upper["learning_rate"]),
            jnp.atleast_1d(config_upper["incentive_reg_param"]),
            jnp.atleast_1d(config["lower_optimisation"]["reg_lambda_decay"]),
        )  # Shape: (n_lr, n_incentive_reg)
        lr_grid = jnp.repeat(
            lr_grid.ravel(), config["num_seeds"]
        )  # Shape: (n_grid_points * num_seeds,)
        incentive_reg_grid = jnp.repeat(
            incentive_reg_grid.ravel(), config["num_seeds"]
        )  # Shape: (n_grid_points * num_seeds,)
        reg_lambda_decay_grid = jnp.repeat(
            lambda_decay_grid.ravel(), config["num_seeds"]
        )  # Shape: (n_grid_points * num_seeds,)
        carry_out, outputs = jax.block_until_ready(
            jax.jit(jax.vmap(run_experiment, in_axes=0))(
                jax.random.split(rng, lr_grid.shape[0]),
                lr_grid,
                incentive_reg_grid,
                reg_lambda_decay_grid,
            )
        )
    else:
        lr_grid = None
        incentive_reg_grid = None
        reg_lambda_decay_grid = None
        carry_out, outputs = jax.block_until_ready(
            jax.jit(jax.vmap(run_experiment, in_axes=(0, None, None, None)))(
                jax.random.split(rng, config["num_seeds"]),
                config_upper["learning_rate"],
                config_upper["incentive_reg_param"],
                config["lower_optimisation"]["reg_lambda_decay"],
            )
        )
    run_time = time.time() - start_time
    print(
        f"Experiment runtime: {(run_time) / 60:.2f} minutes and {(run_time) % 60:.2f} seconds"
    )
    _, env_params, incentive_train_state = carry_out

    # Save results
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    orbax_checkpointer.save(
        os.path.join(
            os.path.abspath(experiment_dir),
            f"checkpoint_incentive_{config['upper_optimisation']['advantage_gradient_sampling']}",
        ),
        incentive_train_state,
        save_args=orbax_utils.save_args_from_target(incentive_train_state),
        force=True,
    )

    with open(
        os.path.join(
            experiment_dir,
            f"metrics_{config['upper_optimisation']['advantage_gradient_sampling']}.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(outputs, f)

    if lr_grid is not None:
        with open(os.path.join(experiment_dir, "grid_search.pkl"), "wb") as f:
            pickle.dump(
                {
                    "lr_grid": lr_grid,
                    "incentive_reg_grid": incentive_reg_grid,
                    "reg_lambda_decay": reg_lambda_decay_grid,
                },
                f,
            )

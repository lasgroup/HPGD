import jax
import jax.numpy as jnp
import yaml
import argparse
import os
import pickle
from typing import Iterable, Tuple, Any, Dict, Callable, Optional
import orbax
from flax.training import orbax_utils
import time
import distrax
from flax.training.train_state import TrainState
from copy import deepcopy
import flax

from src.environments.TaxDesign import (
    TaxDesign,
    EnvParams,
    EnvState,
)
from src.train.Regularized_DQN import create_train, update_dictionary
from src.train.Regularized_DQN import Transition
from src.models.StaticModel import create_state_model as create_static_train_state
from train.utils import update_nested_pytree, remove_non_list_entries
from src.models.ValueNetwork import mse
from src.models.ValueNetwork import create_train_state as create_train_state_value_model


def update_tax_params(
    params: EnvParams,
    train_state_dict: Dict[str, TrainState],
) -> EnvParams:
    """
    Update the tax parameters with the current values from the training state
    """
    return params.replace(
        reward_params=params.reward_params.replace(
            consumption_tax_rate=train_state_dict["vat"].apply_fn(
                train_state_dict["vat"].params
            ),
        ),
        transition_params=params.transition_params.replace(
            income_tax_rate=train_state_dict["income_tax"].apply_fn(
                train_state_dict["income_tax"].params
            )[0],
        ),
    )


def setup_environment(config_setup: dict) -> Tuple[TaxDesign, EnvParams]:
    """
    Initialize the environment
    :param config_setup: Configuration dictionary
    :return: Environment and parameters
    """
    config_upper_optimisation = config_setup["upper_optimisation"]
    env = TaxDesign(
        n_goods=len(config_setup["environment"]["params"]["reward_params"]["prices"]),
        accumulated_asset_utility=lambda x, scale: jnp.where(
            x > 0,
            scale * jnp.log(x / 20 + 1),
            x,
        ),
        max_consumption_tax=config_upper_optimisation["model_params"]["scale"][1],
        max_income_tax=config_upper_optimisation["model_params"]["scale"][1],
        action_discretization=config_setup["environment"]["action_discretization"],
    )
    params = env.default_params
    config_env_params = config_setup["environment"]["params"]
    params = params.replace(
        **{
            key: jax.tree_map(lambda x: jnp.array(x), value)
            for key, value in config_env_params.items()
            if key not in ["reward_params", "transition_params"]
        },
        reward_params=params.reward_params.replace(
            **{
                key: jnp.array(value) if isinstance(value, Iterable) else value
                for key, value in config_env_params["reward_params"].items()
            }
        ),
        transition_params=params.transition_params.replace(
            **{
                key: jnp.array(value) if isinstance(value, Iterable) else value
                for key, value in config_env_params["transition_params"].items()
            }
        ),
    )
    return env, params


def sample_xi(
    key: jax.random.PRNGKey,
    params: EnvParams,
    consumption_preferences_fixed: jax.Array,
) -> EnvParams:
    """
    Sample a fixed consumption preference from the consumption preferences
    """
    xi_idx = jax.random.choice(key, consumption_preferences_fixed.shape[0])
    new_consumption_preferences = consumption_preferences_fixed[xi_idx]
    return (
        params.replace(
            reward_params=params.reward_params.replace(
                consumption_preferences=new_consumption_preferences
            )
        ),
        xi_idx,
    )


def create_trajectory_batch_sample(
    config_create: Dict,
    env: TaxDesign,
    env_params: EnvParams,
) -> Callable[[jax.random.PRNGKey, TrainState, EnvParams, Optional[float]], Transition]:
    """
    Create the trajectory batch sampling function
    :param config_create:
    :param env:
    :param env_params:
    :return:
    """
    config_lower_optimisation = config_create["lower_optimisation"]
    config_lower_training = config_lower_optimisation["training"]
    config_upper_optimisation = config_create["upper_optimisation"]

    vmap_reset = lambda n_envs: lambda rng, params: jax.vmap(
        env.reset, in_axes=(0, None)
    )(jax.random.split(rng, n_envs), params)
    vmap_step = lambda n_envs: lambda rng, env_state, action, params: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, params)
    action_space_shape = [s.n for s in env.action_space(env_params).spaces]

    def get_trajectory_batch(
        key: jax.random.PRNGKey,
        lower_level_train_state: TrainState,
        env_params_sampling: EnvParams,
        eps: float = 0.0,
    ):
        q_function_vmap = jax.vmap(
            lambda o: lower_level_train_state.apply_fn(
                lower_level_train_state.params, jnp.atleast_1d(o)
            )
        )

        def rollout_step(carry, unused):
            rng_carry, env_state_carry, last_obs = carry
            rng_carry, rng_a1, rng_a2, rng_eps, rng_s = jax.random.split(rng_carry, 5)
            # Get the action
            q_values = q_function_vmap(
                last_obs
            )  # Shape: (num_envs, num_actions) or Sequence of (num_envs, )
            if isinstance(q_values, list):
                action_greedy = jax.tree_map(
                    lambda q, key: distrax.Categorical(
                        logits=q / config_lower_optimisation["reg_lambda"]
                    ).sample(seed=key),
                    q_values,
                    list(jax.random.split(rng_a1, len(q_values))),
                )
                action_greedy = jnp.stack(
                    action_greedy, -1
                )  # Shape: (num_envs, num_actions)
            else:
                action_greedy = distrax.Categorical(
                    logits=q_values / config_lower_optimisation["reg_lambda"]
                ).sample(
                    seed=rng_a1
                )  # Shape: (num_envs,)
                action_greedy = jnp.unravel_index(action_greedy, action_space_shape)
                action_greedy = jnp.stack(
                    action_greedy, -1
                )  # Shape: (num_envs, num_actions)
            # Random action
            action_random = jax.random.randint(
                rng_a2,
                shape=(),
                minval=0,
                maxval=jnp.prod(jnp.array(action_space_shape)),
            )
            action_random = jnp.unravel_index(action_random, action_space_shape)
            action_random = jnp.stack(
                action_random, -1
            )  # Shape: (num_envs, num_actions)
            action = jnp.where(
                jax.random.bernoulli(rng_eps, eps),
                action_random,
                action_greedy,
            )
            obs, env_state_carry_new, reward, done, info = vmap_step(
                config_lower_training["num_envs"]
            )(rng_s, env_state_carry, action, env_params_sampling)
            transition = Transition(
                obs=last_obs,
                action=action,
                reward=reward,
                done=done,
                state=env_state_carry,
            )

            carry = (rng_carry, env_state_carry_new, obs)
            return carry, transition

        key_init, key_rollout = jax.random.split(key, 2)
        init_obs, init_env_state = vmap_reset(config_lower_training["num_envs"])(
            key_init, env_params_sampling
        )
        _, traj_batch = jax.lax.scan(
            rollout_step,
            (key_rollout, init_env_state, init_obs),
            None,
            config_upper_optimisation["num_estimation_steps"]
            // config_lower_training["num_envs"],
        )
        return traj_batch

    return get_trajectory_batch


def calculate_discounted_rewards(
    reward_function_params,
    reward_function: Callable,
    traj_batch: Transition,
    discount_factor: float,
    initial_value: Any,
) -> Tuple[jax.Array, jax.Array]:
    """
    Calculate the discounted rewards for a trajectory batch
    :param reward_function_params:
    :param reward_function:
    :param traj_batch:
    :param discount_factor:
    :param initial_value: Initial value for the discounted rewards, matching the shape of the output of reward_function
    :return: Discounted rewards, matching the shape of the initial_value
        The returned array has shape (n_steps, num_envs, pyTree structure of initial_value)
    """

    def _get_discounted_reward(
        rolling_discounted_rewards: jax.Array,
        transition: Transition,
    ) -> Tuple[jax.Array, jax.Array]:
        # vmap over num_envs dimension
        reward = jax.vmap(
            reward_function,
            in_axes=(0, 0, None),
        )(
            transition.state, transition.action, reward_function_params
        )  # Shape: (num_envs, Optional[params_dim] )
        done = transition.done.astype(jnp.float32)  # Shape: (num_envs,)
        rolling_discounted_rewards = jax.tree_map(
            lambda x, y: (
                x
                + discount_factor
                * (1 - (done if len(x.shape) == 1 else jnp.atleast_2d(done).T))
                * y
            ),
            reward,
            rolling_discounted_rewards,
        )  # Calculate discounted reward, Shape: (num_envs, Optional[params_dim])
        return rolling_discounted_rewards, (reward, rolling_discounted_rewards)

    _, (rewards, discounted_rewards) = jax.lax.scan(
        _get_discounted_reward,
        initial_value,
        traj_batch,
        reverse=True,
    )  # Shape: (n_steps, num_envs, Optional[params_dim])
    return rewards, discounted_rewards


def social_welfare_gradient(
    env: TaxDesign,
    env_params: EnvParams,
    traj_batch: Transition,
    upper_level_train_states: Dict[str, TrainState],
) -> Tuple[
    jax.Array,
    Tuple[Dict[str, Dict[str, jax.Array]], Dict[str, Dict[str, jax.Array]]],
]:
    """
    Estimate the gradient of the social welfare
    :return: Tuple with the gradient dictionaries for the two parameters
    """

    def social_welfare(
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
        vat_params: jax.Array,
        income_tax_params: jax.Array,
    ) -> jax.Array:
        """Auxiliary function to calculate the social welfare asa function of the tax parameters"""
        vat = upper_level_train_states["vat"].apply_fn(vat_params)
        income_tax = upper_level_train_states["income_tax"].apply_fn(income_tax_params)[
            0
        ]
        params_env_tmp = params.replace(
            reward_params=params.reward_params.replace(
                consumption_tax_rate=vat,
            ),
            transition_params=params.transition_params.replace(
                income_tax_rate=income_tax
            ),
        )
        return env.social_welfare(
            state, action, params_env_tmp
        )

    social_welfare_grad = jax.grad(social_welfare, argnums=[3, 4])
    social_welfare_grad_vmap = jax.vmap(
        jax.vmap(social_welfare_grad, in_axes=(0, 0, None, None, None)),
        in_axes=(0, 0, None, None, None),
    )
    return social_welfare_grad_vmap(
        traj_batch.state,
        traj_batch.action,
        env_params,
        upper_level_train_states["vat"].params,
        upper_level_train_states["income_tax"].params,
    )  # Shape: ((n_steps, num_envs, 3), (n_steps, num_envs, 1))


def estimate_value_function(
    X: jax.Array,
    X_next: jax.Array,
    rewards: jax.Array,
    value_function_estimator: TrainState,
    num_steps: int,
    discount_factor: float,
    l2_reg: float = 0.0,
):
    """
    Estimate the value function from the trajectory batch
    :param traj_batch:
    :param rewards: array of shape: (n_steps, num_envs, Optional[params_dim])
    :param value_function_estimator: TrainState of the value function estimator
    :param num_steps: Number of training steps
    :param discount_factor: Discount factor
    :param l2_reg: L2 regularization parameter for the MSE
    :return:
    """
    X = X.reshape(
        X.shape[0] * X.shape[1], *X.shape[2:]
    )  # Shape: (n_steps*num_envs, Optional[params_dim])
    X_next = X_next.reshape(
        X_next.shape[0] * X_next.shape[1], *X_next.shape[2:]
    )  # Shape: (n_steps*num_envs, Optional[params_dim])
    if rewards.ndim == 2:
        rewards_reshaped = jnp.expand_dims(rewards, -1)
    else:
        rewards_reshaped = rewards
    rewards_reshaped = rewards_reshaped.reshape(
        rewards_reshaped.shape[0] * rewards_reshaped.shape[1],
        *rewards_reshaped.shape[2:],
    )  # Shape: (n_steps*num_envs, Optional[params_dim])
    rewards_max = jnp.max(
        jnp.abs(rewards_reshaped), axis=0, keepdims=True
    )  # Shape: (Optional[params_dim],)
    rewards_reshaped = rewards_reshaped / rewards_max  # Normalize the rewards

    # Fitting
    mse_grad_fn = jax.value_and_grad(mse)

    def value_network_update(train_state_carry, unused):
        v_next = train_state_carry.apply_fn(
            train_state_carry.params, X_next
        )  # Shape: (n_steps*num_envs, Optional[params_dim])
        target = rewards_reshaped + discount_factor * jax.lax.stop_gradient(v_next)
        loss, grads = mse_grad_fn(
            train_state_carry.params, train_state_carry, X, target, l2_reg
        )
        train_state_carry = train_state_carry.apply_gradients(grads=grads)
        return train_state_carry, loss

    value_model_fitted, losses = jax.lax.scan(
        value_network_update,
        value_function_estimator,
        None,
        length=num_steps,
    )

    # Return estimate values for the trajectory batch
    value_estimate = value_model_fitted.apply_fn(
        value_model_fitted.params, X
    )  # Shape: (n_steps*num_envs, Optional[params_dim])
    value_estimate = (rewards_max * value_estimate).reshape(*rewards.shape)
    return value_model_fitted, value_estimate, losses


def calculate_advantage_gradient(
    env: TaxDesign,
    env_params: EnvParams,
    traj_batch: Transition,
    X: jax.Array,
    X_next: jax.Array,
    upper_level_train_states: Dict[str, TrainState],
    value_function_estimator: TrainState,
    config: Dict,
):
    """Estimate the advantage gradient from the trajectory batch"""
    config_lower_training = config["lower_optimisation"]["training"]
    config_upper_optimisation = config["upper_optimisation"]

    # CALCULATE THE GRADIENT OF THE DISCOUNTED REWARDS W.R.T. THE TAX PARAMETERS
    # (Approximation of the Q-function gradient)
    def reward_func(
        state: EnvState, action: jax.Array, params: EnvParams, vat_params: jax.Array
    ):
        """Auxiliary function to calculate the reward as a function of the tax parameters"""
        vat = upper_level_train_states["vat"].apply_fn(vat_params)
        reward_params_tmp = params.reward_params.replace(
            consumption_tax_rate=vat,
        )
        return env.reward(
            state,
            action,
            reward_params_tmp,
        )

    reward_grad = jax.grad(reward_func, argnums=-1)
    LL_rewards_grad, LL_discounted_rewards_grad = calculate_discounted_rewards(
        env_params,
        lambda s, a, p: jnp.nan_to_num(
            reward_grad(s, a, p, upper_level_train_states["vat"].params)["params"][
                "weights"
            ],
            nan=-0.1,
        ),  # Filling NaNs with zeros, NaNs occur when the action is zero
        traj_batch,
        config_upper_optimisation["discount_factor"],
        initial_value=jnp.zeros((config_lower_training["num_envs"], 3)),
    )  # Shape: (n_steps, num_envs, 3), Might contain NaNs if all values are 0

    value_model_params = config_upper_optimisation["value_model_params"]
    _, LL_discounted_rewards_grad_value_estimate, _ = estimate_value_function(
        X,
        X_next,
        LL_rewards_grad,
        value_function_estimator,
        num_steps=value_model_params["num_training_steps"],
        discount_factor=config_upper_optimisation["discount_factor"],
    )  # Shape: (n_steps, num_envs, 3)
    LL_advantage_grad = (
        LL_discounted_rewards_grad - LL_discounted_rewards_grad_value_estimate
    )  # Shape: (n_steps, num_envs, 3)
    return LL_advantage_grad


def calculate_transition_logprob_gradient(
    env: TaxDesign,
    env_params: EnvParams,
    traj_batch: Transition,
    upper_level_train_states: Dict[str, TrainState],
    grad_clip: float = None,
):
    """
    Calculate the gradient of the transition dynamics log probability
    Assumes truncated normal distribution
    """

    def transition(
        state: EnvState,
        action: jax.Array,
        params_env: EnvParams,
        params_income_tax: jax.Array,
    ):
        """Auxiliary function to calculate the transition as a function of the tax parameters"""
        income_tax = upper_level_train_states["income_tax"].apply_fn(params_income_tax)[
            0
        ]
        params_env_tmp = params_env.replace(
            transition_params=params_env.transition_params.replace(
                income_tax_rate=income_tax
            )
        )
        return env.transition(state, action, params_env_tmp.transition_params).assets

    def transition_logprob(
        state: EnvState,
        action: jax.Array,
        new_state: EnvState,
        params_env: EnvParams,
        params_income_tax: jax.Array,
    ):
        """Auxiliary function to calculate the transition log probability as a function of the tax parameters"""
        mean = transition(state, action, params_env, params_income_tax)
        std = params_env.transition_params.transition_std
        lower_bound, upper_bound = params_env.transition_params.asset_range
        return jax.scipy.stats.truncnorm.logpdf(
            new_state.assets,
            a=lower_bound,
            b=upper_bound,
            loc=mean,
            scale=std,
        )

    transition_grad_f = jax.grad(transition, argnums=-1)
    transition_logprob_grad_f = jax.grad(transition_logprob, argnums=-1)
    # Forward-shift the trajectory batch, add NaNs where the time is 0 (i.e. no previous state)
    traj_batch_back_shift = jax.tree_map(
        lambda x: jnp.where(
            jnp.expand_dims(traj_batch.state.time == 0, -1)
            if len(x.shape) > 2
            else traj_batch.state.time == 0,
            jnp.nan,
            jnp.roll(x, shift=1, axis=0),
        ),
        traj_batch,
    )

    transition_grads = jax.vmap(
        jax.vmap(
            lambda s, a, s_next, p: transition_grad_f(
                s,
                a,
                p,
                upper_level_train_states["income_tax"].params,
            ),
            in_axes=(0, 0, 0, None),
        ),
        in_axes=(0, 0, 0, None),
    )(
        traj_batch_back_shift.state,
        traj_batch_back_shift.action,
        traj_batch.state,
        env_params,
    )
    transition_logprob_grads = jax.vmap(
        jax.vmap(
            lambda s, a, s_next, p: transition_logprob_grad_f(
                s,
                a,
                s_next,
                p,
                upper_level_train_states["income_tax"].params,
            ),
            in_axes=(0, 0, 0, None),
        ),
        in_axes=(0, 0, 0, None),
    )(
        traj_batch_back_shift.state,
        traj_batch_back_shift.action,
        traj_batch.state,
        env_params,
    )
    grads = jax.tree_map(
        lambda x, y: jax.lax.select(
            env_params.transition_params.transition_std > 0, x, y
        ),
        transition_logprob_grads,
        transition_grads,
    )
    grads = jax.tree_map(
        lambda x: jnp.clip(x, a_min=-grad_clip, a_max=grad_clip), grads
    )
    return grads  # Shape: ((n_steps, num_envs, 3), (n_steps, num_envs, 1))


def create_update_step(
    env: TaxDesign,
    env_params_create: EnvParams,
    config: Dict,
) -> Callable:
    config_lower_training = config["lower_optimisation"]["training"]
    config_upper_optimisation = config["upper_optimisation"]
    lower_level_train = create_train(
        env,
        env_params_create,
        config["lower_optimisation"],
        return_transition=False,
    )
    get_trajectory_batch = create_trajectory_batch_sample(
        config,
        env,
        env_params_create,
    )

    def update_step(carry, step_input):
        (
            rng_carry,
            env_params_train_carry,
            upper_level_train_states_carry,
            value_function_estimators_carry,
            consumption_preferences_fixed,
        ) = carry
        t, xi_idx = step_input

        # Sample the consumption preference
        env_params_fixed_xi = env_params_train_carry.replace(
            reward_params=env_params_train_carry.reward_params.replace(
                consumption_preferences=consumption_preferences_fixed[xi_idx]
            )
        )

        # Fit the lower-level
        rng_carry, _rng = jax.random.split(rng_carry)
        train_outputs = lower_level_train(
            _rng,
            env_params_fixed_xi,
            None,
        )
        rng_carry, _rng = jax.random.split(rng_carry)
        traj_batch = get_trajectory_batch(
            _rng,
            train_outputs["runner_state"][0],
            env_params_fixed_xi,
            0.0,  # Epsilon greedy parameter
        )  # Shape: (n_steps, num_envs, PyTree Structure of Transition)

        # Calculate the discounted social welfare for the trajectory batch
        social_welfare, social_welfare_discounted = calculate_discounted_rewards(
            env_params_fixed_xi,
            env.social_welfare,
            traj_batch,
            config_upper_optimisation["discount_factor"],
            initial_value=jnp.zeros((config_lower_training["num_envs"],)),
        )  # Shape: (n_steps, num_envs)

        # GRADIENT ESTIMATION
        # Upper-level reward gradient
        social_welfare_grad = social_welfare_gradient(
            env,
            env_params_fixed_xi,
            traj_batch,
            upper_level_train_states_carry,
        )  # Shape: ((n_steps, num_envs, 3), (n_steps, num_envs, 1))

        # Lower-level advantage gradient
        rng_carry, _rng = jax.random.split(rng_carry)

        # Data preparation
        if config["upper_optimisation"]["value_model_params"]["use_time"]:
            X = jnp.stack([traj_batch.obs, traj_batch.state.time], axis=-1)
        else:
            X = jnp.expand_dims(traj_batch.obs, -1)
        X_next = jnp.where(
            jnp.expand_dims(traj_batch.done, -1),
            jnp.full_like(X, jnp.nan),
            jnp.roll(X, shift=-1, axis=0),
        )
        LL_advantage_grad = calculate_advantage_gradient(
            env,
            env_params_fixed_xi,
            traj_batch,
            X,
            X_next,
            upper_level_train_states_carry,
            value_function_estimators_carry["LL_vat_grad"],
            config,
        )  # Shape: (n_steps, num_envs, 3)
        social_welfare_discounted_normalized = social_welfare_discounted
        vat_br_grad = (
            LL_advantage_grad
            * jnp.expand_dims(social_welfare_discounted_normalized, -1)
            / config["lower_optimisation"]["reg_lambda"]
        )  # Shape: (n_steps, num_envs, 3)

        # Social Welfare Value Estimate
        rng_carry, _rng = jax.random.split(rng_carry)
        value_model_params = config_upper_optimisation["value_model_params"]

        _, social_welfare_value_estimate, _ = estimate_value_function(
            X,
            X_next,
            social_welfare,
            value_function_estimators_carry["social_welfare"],
            num_steps=value_model_params["num_training_steps"],
            discount_factor=config_upper_optimisation["discount_factor"],
        )  # Shape: (n_steps, num_envs)

        # Transition dynamics log prob gradient w.r.t. the income tax parameters
        transition_logprob_grad = calculate_transition_logprob_gradient(
            env,
            env_params_fixed_xi,
            traj_batch,
            upper_level_train_states_carry,
            grad_clip=config_upper_optimisation["transition_logprob_grad_clip"],
        )  # Shape: (n_steps, num_envs, 1)

        # Collect Gradients
        traj_batch_discounting = jnp.power(
            config_upper_optimisation["discount_factor"], traj_batch.state.time
        )  # Shape: (n_steps, num_envs)

        # VAT Grad
        vat_sw_grad = social_welfare_grad[0]["params"]["weights"]
        vat_grad = vat_sw_grad + vat_br_grad  # Shape: (n_steps, num_envs, 3)
        num_episodes = jnp.sum(traj_batch.done)  # Shape: ()
        vat_grad = (
            jnp.nansum(
                vat_grad * jnp.expand_dims(traj_batch_discounting, -1), axis=(0, 1)
            )
            / num_episodes
        )  # Shape: (3,)

        # Income Tax Grad
        income_sw_grad = social_welfare_grad[1]["params"]["weights"]
        social_welfare_value_estimate_normalized = social_welfare_value_estimate
        income_transition_grad = transition_logprob_grad["params"][
            "weights"
        ] * jnp.expand_dims(
            social_welfare_value_estimate_normalized, -1
        )
        income_tax_grad = (
            income_sw_grad + income_transition_grad
        )  # Shape: (n_steps, num_envs, 1)
        income_tax_grad = (
            jnp.nansum(
                income_tax_grad * jnp.expand_dims(traj_batch_discounting, -1),
                axis=(0, 1),
            )
            / num_episodes
        )  # Shape: (1,)
        grad = {
            "vat": {"params": {"weights": -vat_grad}},
            "income_tax": {"params": {"weights": -income_tax_grad}},
        }

        # Update the upper-level training states
        upper_level_train_states_carry = {
            key: ts.apply_gradients(
                grads=flax.core.FrozenDict(grad[key])
                if jax.__version__ == "0.4.10"
                else grad[key],
            )
            for key, ts in upper_level_train_states_carry.items()
        }

        # Output metrics
        V_UL = jnp.where(
            traj_batch.state.time == 0,
            social_welfare_discounted,
            jnp.nan,
        )
        V_UL = jnp.nanmean(V_UL)
        discounting_arr = jnp.power(
            config["lower_optimisation"]["discount_factor"],
            traj_batch.state.time,
        )
        V_LL = jnp.sum(traj_batch.reward * discounting_arr) / num_episodes
        episode_length = config["environment"]["params"]["max_steps_in_episode"]
        training_metrics = train_outputs["metrics"]
        metrics = {
            "V_UL": V_UL,
            "V_LL": V_LL,
            "vat": env_params_train_carry.reward_params.consumption_tax_rate,
            "income_tax": env_params_train_carry.transition_params.income_tax_rate,
            "vat_grad": vat_grad,
            "income_tax_grad": income_tax_grad,
            "vat_sw_grad_mean": jnp.mean(vat_sw_grad, axis=(0, 1)),
            "vat_br_grad_mean": jnp.mean(vat_br_grad, axis=(0, 1)),
            "income_sw_grad_mean": jnp.mean(income_sw_grad, axis=(0, 1)),
            "income_transition_grad_mean": jnp.nanmean(
                income_transition_grad, axis=(0, 1)
            ),
            "traj_batch_last_episode_obs": jnp.sum(
                jnp.mean(traj_batch.obs[-episode_length:], 1), 0
            ),  # Mean over num_envs, Sum over time
            "traj_batch_last_episode_actions": jnp.sum(
                jnp.mean(traj_batch.action[-episode_length:], 1), 0
            ),  # Mean over num_envs, Sum over time
            "LL_last_episode_obs": jnp.sum(
                training_metrics["obs"][-episode_length:], 0
            ),
            "LL_last_episode_action": jnp.sum(
                training_metrics["action"][-episode_length:], 0
            ),
        }
        env_params_train_carry = update_tax_params(
            env_params_train_carry, upper_level_train_states_carry
        )
        return (
            rng_carry,
            env_params_train_carry,
            upper_level_train_states_carry,
            value_function_estimators_carry,
            consumption_preferences_fixed,
        ), metrics

    return update_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dir", type=str, help="Path to the experiment directory"
    )
    args = parser.parse_args()
    experiment_dir = args.experiment_dir
    print("Output directory: ", experiment_dir)
    print("Device used: ", jax.devices())
    print("Number of devices: ", jax.local_device_count())

    # Read config
    with open(f"{experiment_dir}/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("Config: ", config)
    rng = jax.random.PRNGKey(config["random_seed"])
    config_init = deepcopy(config)
    config_init["environment"]["params"]["reward_params"][
        "consumption_preferences"
    ] = config["environment"]["params"]["reward_params"]["consumption_preferences"][0]

    # Create the update dictionary
    update_dict = remove_non_list_entries(
        config,
        list_parameters=[
            "asset_range",
            "prices",
            "consumption_tax_rate",
            "hidden_layers",
            "scale",
            "layer_size",
        ],
        omit_parameters=["consumption_preferences"],
    )
    update_dict = jax.tree_map(
        lambda x: jnp.array(x), update_dict, is_leaf=lambda x: isinstance(x, list)
    )
    leaves, tree_structure = jax.tree_util.tree_flatten(
        update_dict, is_leaf=lambda x: isinstance(x, jax.Array)
    )
    leaves_idx = [jnp.arange(len(leaf)) for leaf in leaves]
    meshgrid = jnp.meshgrid(*leaves_idx)
    update_dict = jax.tree_map(
        lambda idx, x: x[idx.reshape(-1), ...],
        jax.tree_util.tree_unflatten(tree_structure, meshgrid),
        update_dict,
    )
    print("Update dict: ", update_dict)

    # Create environment
    basic_env, basic_env_params = setup_environment(config_init)
    print("Basic Env params: ", basic_env_params)
    consumption_preferences_fixed = jnp.array(
        config["environment"]["params"]["reward_params"]["consumption_preferences"]
    )

    def run_experiment(
        key: jax.random.PRNGKey,
        config_update: Dict[str, Any],
        env_params_exp: EnvParams,
    ) -> Tuple[
        Tuple[jax.Array, EnvParams, Dict[str, TrainState], jax.Array],
        Tuple[jax.Array],
    ]:
        config_exp = deepcopy(config_init)
        config_exp = update_dictionary(config_exp, config_update)
        env_params_exp = update_nested_pytree(
            env_params_exp, config_exp["environment"]["params"]
        )

        # Initialize the upper level
        config_upper_optimisation_model = config_exp["upper_optimisation"][
            "model_params"
        ]
        key, _rng1, _rng2 = jax.random.split(key, 3)
        upper_level_train_states = {
            "vat": create_static_train_state(
                param_shape=(3,),
                key=_rng1,
                init_value=jnp.array(env_params_exp.reward_params.consumption_tax_rate),
                **config_upper_optimisation_model,
            ),
            "income_tax": create_static_train_state(
                param_shape=(1,),
                key=_rng2,
                init_value=jnp.atleast_1d(env_params_exp.transition_params.income_tax_rate),
                **config_upper_optimisation_model,
            ),
        }
        # Initialize the value function estimators
        key, _rng1, _rng2 = jax.random.split(key, 3)
        config_value_model = config_exp["upper_optimisation"]["value_model_params"]
        value_function_estimators = {
            "LL_vat_grad": create_train_state_value_model(
                key=_rng1,
                input_dim=2 if config_value_model["use_time"] else 1,
                output_dim=3,
                layer_size=[
                    64,
                ],  # TODO: remove hardcode
                optimizer_params=config_value_model["optimizer_params"],
            ),
            "social_welfare": create_train_state_value_model(
                key=_rng1,
                input_dim=2 if config_value_model["use_time"] else 1,
                output_dim=1,
                layer_size=[
                    64,
                ],
                optimizer_params=config_value_model["optimizer_params"],
            ),
        }

        # TRAINING
        update_step = create_update_step(basic_env, env_params_exp, config_exp)
        n_iter = config_exp["upper_optimisation"]["num_outer_iter"]
        time_array = jnp.arange(1, n_iter + 1)
        key, _rng = jax.random.split(key)
        xi_idx_arr = jax.random.choice(
            _rng, consumption_preferences_fixed.shape[0], shape=(n_iter,)
        )
        carry, metrics = jax.lax.scan(
            update_step,
            (
                key,
                env_params_exp,
                upper_level_train_states,
                value_function_estimators,
                consumption_preferences_fixed,
            ),
            (time_array, xi_idx_arr),
            n_iter,
        )
        metrics["xi_idx"] = xi_idx_arr
        return carry, metrics

    # RUN EXPERIMENT
    start_time = time.time()
    if len(update_dict) > 0:
        run_experiment_vmap = jax.vmap(
            jax.vmap(
                run_experiment,
                in_axes=(None, jax.tree_map(lambda x: 0, update_dict), None),
            ),
            in_axes=(0, None, None),
        )
        carry_out, output_metrics = jax.block_until_ready(
            jax.jit(run_experiment_vmap)(
                jax.random.split(rng, config_init["num_seeds"]),
                update_dict,
                basic_env_params,
            )
        )
    else:
        run_experiment_vmap = jax.vmap(run_experiment, in_axes=(0, None, None))
        carry_out, output_metrics = jax.block_until_ready(
            jax.jit(run_experiment_vmap)(
                jax.random.split(rng, config_init["num_seeds"]),
                update_dict,
                basic_env_params,
            )
        )
    run_time = time.time() - start_time
    print(
        f"Experiment runtime: {(run_time) / 60:.2f} minutes and {(run_time) % 60:.2f} seconds"
    )
    _, env_params_out, upper_level_train_states_out, _, _ = carry_out

    # SAVE RESULTS
    with open(os.path.join(experiment_dir, "metrics_hpgd.pkl"), "wb") as f:
        pickle.dump(output_metrics, f)
    with open(os.path.join(experiment_dir, "update_dict_hpgd.pkl"), "wb") as f:
        pickle.dump(update_dict, f)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(upper_level_train_states_out)
    orbax_checkpointer.save(
        os.path.join(os.path.abspath(experiment_dir), "checkpoint_incentive_hpgd"),
        upper_level_train_states_out,
        force=True,
    )

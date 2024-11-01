import argparse
import os

import flax.core.frozen_dict
import yaml
import pickle
import time
from typing import Dict, Tuple, Callable, NamedTuple, Iterable

import jax
import jax.numpy as jnp
import orbax
from flax.training import orbax_utils
from flax.training.train_state import TrainState

from src.environments.ConfigurableFourRooms import ConfigurableFourRooms
from src.environments.utils import sample_array
from src.models.IncentiveModel import create_incentive_train_state
from src.algorithms.value_iteration_and_prediction import (
    general_value_iteration,
    get_reward_matrix,
)
from train_stochastic_bilevel_opt import (
    environment_setup,
    upper_level_reward,
    regularized_softmax,
)


def update_value_functions(
    Q_LL_carry,
    Q_LL_grad_carry,
    policy_carry,
    config_lower_level,
):
    # Update value functions
    V_LL = jnp.sum(policy_carry * Q_LL_carry, axis=-1) - config_lower_level[
        "reg_lambda"
    ] * jnp.sum(
        policy_carry * jnp.log(policy_carry + 1e-16), axis=-1
    )  # Shape: (n_goals, n_states)

    # Update the other value functions
    V_LL_grad = jnp.sum(
        jnp.expand_dims(policy_carry, 3) * Q_LL_grad_carry, axis=2
    )  # Shape: (n_goals, n_states, |params|)
    Advantage_LL = Q_LL_carry - jnp.expand_dims(
        V_LL, 2
    )  # Shape: (n_goals, n_states, n_actions)
    Advantage_LL_grad = Q_LL_grad_carry - jnp.expand_dims(
        V_LL_grad, 2
    )  # Shape: (n_goals, n_states, n_actions, |params|)
    return V_LL, V_LL_grad, Advantage_LL, Advantage_LL_grad


def initializations(
    env: ConfigurableFourRooms,
    env_params: NamedTuple,
    env_reward_grad: Callable,
    config: Dict,
):
    config_lower_level = config["lower_optimisation"]
    uniform_policy = (
        jnp.ones((len(env.available_goals), env.coords.shape[0], env.num_actions))
        / env.num_actions
    )

    Q_LL, _ = general_value_iteration(
        env,
        env_params,
        gamma=config_lower_level["discount_factor"],
        n_policy_iter=config_lower_level["n_policy_iter"],
        n_value_iter=config_lower_level["n_value_iter"],
        regularization=config_lower_level["regularization"],
        reg_lambda=config_lower_level["reg_lambda"],
        return_q_value=True,
    )

    Q_LL_grad, _ = general_value_iteration(
        env,
        env_params,
        gamma=config_lower_level["discount_factor"],
        n_policy_iter=config_lower_level["n_policy_iter"],
        n_value_iter=config_lower_level["n_value_iter"],
        regularization=config_lower_level["regularization"],
        reg_lambda=config_lower_level["reg_lambda"],
        return_q_value=True,
        policy=uniform_policy,
        external_reward=env_reward_grad,
    )

    Q_UL, _ = general_value_iteration(
        env,
        env_params,
        gamma=config_lower_level["discount_factor"],
        n_policy_iter=config_lower_level["n_policy_iter"],
        n_value_iter=config_lower_level["n_value_iter"],
        return_q_value=True,
        policy=uniform_policy,
        external_reward=lambda s, a, env_p: upper_level_reward(
            env_p.incentive_params, s, a, config
        ),
    )

    V_LL, V_LL_grad, Advantage_LL, Advantage_LL_grad = update_value_functions(
        Q_LL, Q_LL_grad, uniform_policy, config_lower_level
    )

    def grad_reward(state, action, env_p):
        goal_mask = jnp.expand_dims(
            jnp.all(env.available_goals == state.goal, 1), (1, 2)
        )  # Shape: (n_goals, 1, 1)
        state_mask = jnp.expand_dims(
            jnp.all(env.coords == state.pos, 1), (0, 2)
        )  # Shape: (1, n_states, 1)
        action_mask = jnp.expand_dims(
            jax.nn.one_hot(action, env.num_actions), (0, 1)
        )  # Shape: (1, 1, n_actions)
        combined_mask = (
            goal_mask * state_mask * action_mask
        )  # Shape: (n_goals, n_states, n_actions)
        reward_grad = env_reward_grad(state, action, env_p)  # Shape: (|params|)
        LL_advantage_term = (
            jnp.expand_dims(jnp.sum(combined_mask * Advantage_LL), -1)
            * jnp.sum(combined_mask[..., None] * Advantage_LL_grad, axis=(0, 1, 2))
        ) / config["lower_optimisation"][
            "reg_lambda"
        ]  # Shape: (|params|)
        return reward_grad + LL_advantage_term

    Q_tilde, _ = general_value_iteration(
        env,
        env_params,
        gamma=config_lower_level["discount_factor"],
        n_policy_iter=config_lower_level["n_policy_iter"],
        n_value_iter=config_lower_level["n_value_iter"],
        regularization=config_lower_level["regularization"],
        reg_lambda=config_lower_level["reg_lambda"],
        return_q_value=True,
        policy=uniform_policy,
        external_reward=grad_reward,
    )
    return (
        Q_LL,
        Q_LL_grad,
        Q_UL,
        V_LL,
        V_LL_grad,
        Advantage_LL,
        Advantage_LL_grad,
        Q_tilde,
    )


def create_update_step(
    env: ConfigurableFourRooms,
    config: Dict,
) -> Callable:
    config_lower_level = config["lower_optimisation"]

    is_terminal_state = jnp.expand_dims(
        env.terminal_states, -1
    )  # Shape: (len(env.available_goals), n_states, 1)

    def env_reward_grad(state, action, params):
        f_grad = jax.grad(
            lambda s, a, p: env.get_reward(s, a, params.replace(incentive_params=p)),
            argnums=2,
        )
        return f_grad(state, action, params.incentive_params)["params"][
            "weights"
        ]  # Shape: (|params|)

    def update_step(carry_update, reg_lambda):
        (
            rng,
            env_params_train_carry,
            incentive_train_state_carry,
            lower_level_iter_carry,
        ) = carry_update

        transition_probability_matrix = env.get_transition_probability_matrix(
            env_params_train_carry
        )  # Shape: (len(env.available_goals), n_states, n_actions, n_states)
        reward_matrix = get_reward_matrix(
            env, env_params_train_carry
        )  # Shape: (len(env.available_goals), n_states, n_actions)
        reward_matrix_grad = get_reward_matrix(
            env, env_params_train_carry, external_reward=env_reward_grad
        )  # Shape: (len(env.available_goals), n_states, n_actions, |params|)
        reward_matrix_UL = get_reward_matrix(
            env,
            env_params_train_carry,
            external_reward=lambda s, a, env_p: upper_level_reward(
                env_p.incentive_params, s, a, config
            ),
        )  # Shape: (len(env.available_goals), n_states, n_actions)

        def UL_reward_grad(state, action, params):
            return jax.grad(
                lambda s, a, p: upper_level_reward(p, s, a, config), argnums=2
            )(state, action, params.incentive_params)["params"]["weights"]

        reward_matrix_UL_grad = get_reward_matrix(
            env,
            env_params_train_carry,
            external_reward=UL_reward_grad,
        )  # Shape: (len(env.available_goals), n_states, n_actions, |params|)

        rng, _rng = jax.random.split(rng)
        _, xi_idx, goal_probs = sample_array(
            _rng, env.available_goals, env_params_train_carry.resample_goal_logits
        )

        def update_values(carry, unused):
            Q_LL_carry, Q_LL_grad_carry, Q_UL_carry, Q_tilde_carry = carry
            # Update Policy
            policy = regularized_softmax(
                Q_LL_carry, reg_lambda
            )  # Shape: (n_goals, n_states, n_actions)

            # Update value functions
            V_LL, V_LL_grad, Advantage_LL, Advantage_LL_grad = update_value_functions(
                Q_LL_carry, Q_LL_grad_carry, policy, config_lower_level
            )
            V_UL = jnp.sum(policy * Q_UL_carry, axis=-1)  # Shape: (n_goals, n_states)
            Advantage_UL = Q_UL_carry - jnp.expand_dims(
                V_UL, 2
            )  # Shape: (n_goals, n_states, n_actions)
            V_tilde = jnp.sum(
                jnp.expand_dims(policy, 3) * Q_tilde_carry, axis=2
            )  # Shape: (n_goals, n_states, |params|)

            # Update Q-values for the lower-level
            discounted_next_value = config_lower_level["discount_factor"] * jnp.sum(
                transition_probability_matrix * jnp.expand_dims(V_LL, (1, 2)), axis=3
            )  # Shape: (n_goals, n_states, n_actions)
            Q_LL_new = reward_matrix + jnp.where(
                is_terminal_state, 0.0, discounted_next_value
            )  # Shape: (n_goals, n_states, n_actions)
            Q_error = jnp.mean(jnp.abs(Q_LL_new - Q_LL_carry))

            # Update gradient Q-values
            discounted_next_value = config_lower_level["discount_factor"] * jnp.sum(
                jnp.expand_dims(transition_probability_matrix, -1)
                * jnp.expand_dims(V_LL_grad, (1, 2)),
                axis=3,
            )  # Shape: (n_goals, n_states, n_actions, |params|)
            Q_LL_grad_carry = reward_matrix_grad + jnp.where(
                jnp.expand_dims(is_terminal_state, -1), 0.0, discounted_next_value
            )  # Shape: (n_goals, n_states, n_actions, |params|)

            # Update Q-values for the upper-level
            discounted_next_value = config["upper_optimisation"][
                "discount_factor"
            ] * jnp.sum(
                transition_probability_matrix * jnp.expand_dims(V_UL, (1, 2)), axis=3
            )  # Shape: (n_goals, n_states, n_actions)
            Q_UL_carry = reward_matrix_UL + jnp.where(
                is_terminal_state, 0.0, discounted_next_value
            )  # Shape: (n_goals, n_states, n_actions)

            # Update gradient reward Q-values
            discounted_next_value = config_lower_level["discount_factor"] * jnp.sum(
                jnp.expand_dims(transition_probability_matrix, -1)
                * jnp.expand_dims(V_tilde, (1, 2)),
                axis=3,
            )  # Shape: (n_goals, n_states, n_actions, |params|)
            Q_tilde_carry = (
                reward_matrix_UL_grad
                + Advantage_LL_grad
                * jnp.expand_dims(Advantage_UL, -1)
                / reg_lambda  # Shape: (n_goals, n_states, n_actions, |params|)
                + jnp.where(
                    jnp.expand_dims(is_terminal_state, -1), 0.0, discounted_next_value
                )
            )
            return (Q_LL_new, Q_LL_grad_carry, Q_UL_carry, Q_tilde_carry), (Q_error,)

        lower_level_iter_carry, lower_level_outputs = jax.lax.scan(
            update_values,
            lower_level_iter_carry,
            None,
            length=config_lower_level["n_policy_iter"],
        )
        Q_LL, _, Q_UL, Q_tilde = lower_level_iter_carry
        policy = regularized_softmax(
            Q_LL, reg_lambda
        )  # Shape: (n_goals, n_states, n_actions)
        V_tilde = jnp.sum(
            jnp.expand_dims(policy, 3) * Q_tilde, axis=2
        )  # Shape: (n_goals, n_states, |params|)

        init_state_probs = env.state_initialization_distribution(
            env_params_train_carry.state_initialization_params
        ).probs
        init_position_idx = jnp.all(
            env.coords[..., None] == env.available_init_pos.T[None, ...], axis=1
        )  # Shape: (n_states, n_init_pos)
        grad = jnp.sum(
            jnp.sum(jnp.expand_dims(init_state_probs, 0) * init_position_idx, axis=1)[
                ..., None
            ]
            * V_tilde[xi_idx, :, :],
            0,
        )  # Shape: (|params|)
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
        V_UL = jnp.sum(policy * Q_UL, axis=-1)  # Shape: (n_goals, n_states)
        UL_initial_value = jnp.sum(
            jnp.sum(jnp.expand_dims(init_state_probs, 0) * init_position_idx, axis=1)[
                None, ...
            ]
            * V_UL,
            axis=1,
        )  # Shape: (n_goals)
        UL_initial_value = jnp.sum(goal_probs * UL_initial_value)
        return (
            rng,
            env_params_train_carry,
            incentive_train_state_carry,
            lower_level_iter_carry,
        ), {
            "Q_error": lower_level_outputs[0],
            "UL_initial_value": UL_initial_value,
            "grad_norm": jnp.linalg.norm(grad),
        }

    return update_step


if __name__ == "__main__":
    # experiment_dir = "data/results/budget_0_2"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dir", type=str, help="Path to the experiment directory"
    )
    args = parser.parse_args()
    experiment_dir = args.experiment_dir
    print("Output directory: ", experiment_dir)
    print("Device used: ", jax.devices())

    config = yaml.safe_load(open(os.path.join(experiment_dir, "config.yaml"), "r"))
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
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, TrainState, Tuple], Tuple[jnp.ndarray]]:
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

        def env_reward_grad(state, action, params):
            f_grad = jax.grad(
                lambda s, a, p: env.get_reward(
                    s, a, params.replace(incentive_params=p)
                ),
                argnums=2,
            )
            return f_grad(state, action, params.incentive_params)["params"][
                "weights"
            ]  # Shape: (|params|)

        # Lower level initializations
        (
            Q_LL,
            Q_LL_grad,
            Q_UL,
            V_LL,
            V_LL_grad,
            Advantage_LL,
            Advantage_LL_grad,
            Q_tilde,
        ) = initializations(env, env_params_exp, env_reward_grad, config_exp)

        # TRAINING
        n_iter = config_exp["upper_optimisation"]["num_outer_iter"]
        reg_lambda = config["lower_optimisation"]["reg_lambda"] * jnp.power(lower_optimisation_reg_lambda_decay, jnp.arange(n_iter))
        update_step = create_update_step(env, config_exp)
        lower_level_carry = (Q_LL, Q_LL_grad, Q_UL, Q_tilde)
        return jax.lax.scan(
            update_step,
            (rng, env_params_exp, incentive_train_state_exp, lower_level_carry),
            reg_lambda,
            config_exp["upper_optimisation"]["num_outer_iter"],
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
        lr_grid, incentive_reg_grid, lambda_decay_grid  = jnp.meshgrid(
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
        lambda_decay_grid = jnp.repeat(
            lambda_decay_grid.ravel(), config["num_seeds"]
        )
        carry_out, outputs = jax.block_until_ready(
            jax.jit(jax.vmap(run_experiment, in_axes=0))(
                jax.random.split(rng, lr_grid.shape[0]), lr_grid, incentive_reg_grid, lambda_decay_grid
            )
        )
    else:
        lr_grid = None
        incentive_reg_grid = None
        lambda_decay_grid = None
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
    _, env_params, incentive_train_state, lower_level_carry = carry_out

    # Save results
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    orbax_checkpointer.save(
        os.path.join(os.path.abspath(experiment_dir), "checkpoint_incentive_benchmark"),
        incentive_train_state,
        save_args=orbax_utils.save_args_from_target(incentive_train_state),
        force=True,
    )

    with open(os.path.join(experiment_dir, "metrics_benchmark.pkl"), "wb") as f:
        pickle.dump(outputs, f)

    if lr_grid is not None:
        with open(os.path.join(experiment_dir, "grid_search_benchmark.pkl"), "wb") as f:
            pickle.dump(
                {"lr_grid": lr_grid, "incentive_reg_grid": incentive_reg_grid}, f
            )

import argparse
import os

import flax.core.frozen_dict
import yaml
import pickle
import time
from typing import Dict, Tuple, Callable, Iterable

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
    initial_value_prediction,
)
from train_stochastic_bilevel_opt import (
    environment_setup,
    upper_level_reward,
    regularized_softmax,
)


def random_split_like_tree(rng_key, target=None, treedef=None):
    if treedef is None:
        treedef = jax.tree_util.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)


def tree_random_normal_like(rng_key, target):
    keys_tree = random_split_like_tree(rng_key, target)
    return jax.tree_map(
        lambda l, k: jax.random.normal(k, l.shape, l.dtype),
        target,
        keys_tree,
    )


def create_update_step(
    env: ConfigurableFourRooms,
    config: Dict,
) -> Callable:
    config_lower_level = config["lower_optimisation"]
    config_upper_level = config["upper_optimisation"]

    def update_step(carry, step_input):
        rng, env_params_train_carry, incentive_train_state_carry = carry
        t, reg_lambda = step_input

        # Realize Xi
        rng, _rng = jax.random.split(rng)
        _, xi_idx, goal_probs = sample_array(
            _rng, env.available_goals, env_params_train_carry.resample_goal_logits
        )
        env_params_fixed_xi = env_params_train_carry.replace(
            resample_goal_logits=jnp.log(
                jnp.full_like(env_params_train_carry.resample_goal_logits, 1e-16)
                .at[xi_idx]
                .set(1.0)
            )
        )

        # Estimate Value for Xi
        Q_LL, _ = general_value_iteration(
            env,
            env_params_fixed_xi,
            gamma=config_lower_level["discount_factor"],
            n_policy_iter=config_lower_level["n_policy_iter"],
            n_value_iter=config_lower_level["n_value_iter"],
            regularization=config_lower_level["regularization"],
            reg_lambda=reg_lambda,
            return_q_value=True,
        )  # Shape: (n_goals, n_states, n_actions)
        policy = regularized_softmax(Q_LL, reg_lambda)
        V_UL, _ = initial_value_prediction(
            env,
            env_params_fixed_xi,
            gamma=config_upper_level["discount_factor"],
            n_policy_iter=config_lower_level["n_policy_iter"],
            n_value_iter=config_lower_level["n_value_iter"],
            policy=policy[xi_idx],
            external_reward=lambda s, a, env_p: upper_level_reward(
                env_p.incentive_params, s, a, config
            ),
        )

        # Sample Z and u
        rng, _rng = jax.random.split(rng)
        Z = tree_random_normal_like(_rng, env_params_train_carry.incentive_params)
        u = config_upper_level["zero_order_perturbation_constant"] / t
        env_params_tmp = env_params_fixed_xi.replace(
            incentive_params=jax.tree_map(
                lambda x, z: x + u * z, env_params_train_carry.incentive_params, Z
            )
        )
        Q_LL_perturbed, _ = general_value_iteration(
            env,
            env_params_tmp,
            gamma=config_lower_level["discount_factor"],
            n_policy_iter=config_lower_level["n_policy_iter"],
            n_value_iter=config_lower_level["n_value_iter"],
            regularization=config_lower_level["regularization"],
            reg_lambda=reg_lambda,
            return_q_value=True,
        )
        policy_perturbed = regularized_softmax(Q_LL_perturbed, reg_lambda)
        V_UL_perturbed, _ = initial_value_prediction(
            env,
            env_params_tmp,
            gamma=config_upper_level["discount_factor"],
            n_policy_iter=config_lower_level["n_policy_iter"],
            n_value_iter=config_lower_level["n_value_iter"],
            policy=policy_perturbed[xi_idx],
            external_reward=lambda s, a, env_p: upper_level_reward(
                env_p.incentive_params, s, a, config
            ),
        )

        grad = jax.tree_map(
            lambda z: -(V_UL_perturbed - V_UL) * z / u, Z
        )  # Negate because we want to maximize the reward

        incentive_train_state_carry = incentive_train_state_carry.apply_gradients(
            grads=flax.core.frozen_dict.FrozenDict(grad)
            if jax.__version__ == "0.4.10"
            else grad
        )
        env_params_train_carry = env_params_train_carry.replace(
            incentive_params=incentive_train_state_carry.params
        )

        return (
            rng,
            env_params_train_carry,
            incentive_train_state_carry,
        ), {
            "xi_idx": xi_idx,
            "UL_initial_value": V_UL,
            "UL_perturbed_initial_value_estimate": V_UL_perturbed,
            "grad_norm": sum(
                jax.tree_util.tree_leaves(
                    jax.tree_map(lambda x: jnp.linalg.norm(x), grad)
                )
            ),
        }

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
        upper_optimisation_zero_order_perturbation_constant: float,
        lower_optimisation_reg_lambda_decay: float,
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, TrainState], Tuple[jnp.ndarray]]:
        config_exp = config.copy()
        config_exp["upper_optimisation"]["learning_rate"] = upper_optimisation_lr
        config_exp["upper_optimisation"][
            "incentive_reg_param"
        ] = upper_optimisation_incentive_reg
        config_exp["upper_optimisation"][
            "zero_order_perturbation_constant"
        ] = upper_optimisation_zero_order_perturbation_constant

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
            (jnp.arange(1, n_iter + 1), reg_lambda),
            n_iter,
        )

    start_time = time.time()
    config_upper = config["upper_optimisation"]
    if (
        isinstance(config_upper["learning_rate"], Iterable)
        or isinstance(config_upper["incentive_reg_param"], Iterable)
        or isinstance(config_upper["zero_order_perturbation_constant"], Iterable)
        or isinstance(config["lower_optimisation"]["reg_lambda_decay"], Iterable)
    ):
        (
            lr_grid,
            incentive_reg_grid,
            zero_order_perturbation_constant_grid,
            lambda_decay_grid,
        ) = jnp.meshgrid(
            jnp.atleast_1d(config_upper["learning_rate"]),
            jnp.atleast_1d(config_upper["incentive_reg_param"]),
            jnp.atleast_1d(config_upper["zero_order_perturbation_constant"]),
            jnp.atleast_1d(config["lower_optimisation"]["reg_lambda_decay"]),
        )
        lr_grid = jnp.repeat(lr_grid.ravel(), config["num_seeds"])
        incentive_reg_grid = jnp.repeat(incentive_reg_grid.ravel(), config["num_seeds"])
        zero_order_perturbation_constant_grid = jnp.repeat(
            zero_order_perturbation_constant_grid.ravel(), config["num_seeds"]
        )
        lambda_decay_grid = jnp.repeat(lambda_decay_grid.ravel(), config["num_seeds"])
        carry_out, outputs = jax.block_until_ready(
            jax.jit(jax.vmap(run_experiment, in_axes=0))(
                jax.random.split(rng, lr_grid.shape[0]),
                lr_grid,
                incentive_reg_grid,
                zero_order_perturbation_constant_grid,
                lambda_decay_grid,
            )
        )
    else:
        (
            lr_grid,
            incentive_reg_grid,
            zero_order_perturbation_constant_grid,
            lambda_decay_grid,
        ) = (None, None, None, None)
        carry_out, outputs = jax.block_until_ready(
            jax.jit(jax.vmap(run_experiment, in_axes=(0, None, None, None, None)))(
                jax.random.split(rng, config["num_seeds"]),
                config_upper["learning_rate"],
                config_upper["incentive_reg_param"],
                config_upper["zero_order_perturbation_constant"],
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
            os.path.abspath(experiment_dir), "checkpoint_incentive_zero_order"
        ),
        incentive_train_state,
        save_args=orbax_utils.save_args_from_target(incentive_train_state),
        force=True,
    )

    with open(os.path.join(experiment_dir, "metrics_zero_order.pkl"), "wb") as f:
        pickle.dump(outputs, f)

    if lr_grid is not None:
        with open(
            os.path.join(experiment_dir, "grid_search_zero_order.pkl"), "wb"
        ) as f:
            pickle.dump(
                {
                    "lr_grid": lr_grid,
                    "incentive_reg_grid": incentive_reg_grid,
                    "zero_order_perturbation_constant_grid": zero_order_perturbation_constant_grid,
                },
                f,
            )

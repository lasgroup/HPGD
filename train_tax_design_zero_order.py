import flax.core
import jax
import jax.numpy as jnp
import yaml
import argparse
import os
import pickle
from typing import Tuple, Any, Dict, Callable
import orbax
from flax.training import orbax_utils
import time
from train.utils import update_nested_pytree, remove_non_list_entries

from copy import deepcopy

from src.environments.TaxDesign import (
    TaxDesign,
    EnvParams,
)
from src.train.Regularized_DQN import create_train, update_dictionary
from src.models.StaticModel import create_state_model as create_static_train_state
from train_tax_design import (
    setup_environment,
    create_trajectory_batch_sample,
    calculate_discounted_rewards,
    update_tax_params,
)
from flax.training.train_state import TrainState


def random_normal_perturbations(rng_key, train_states):
    key_split = jax.random.split(rng_key, len(train_states))
    keys_tree = {key: key_split[i] for i, key in enumerate(train_states)}
    return {
        key: jax.random.normal(
            keys_tree[key],
            shape=ts.params["params"]["weights"].shape,
            dtype=ts.params["params"]["weights"].dtype,
        )
        for key, ts in train_states.items()
    }


def create_update_step(
    env: TaxDesign,
    env_params: EnvParams,
    config: Dict,
) -> Callable:
    config_lower_training = config["lower_optimisation"]["training"]
    config_upper_optimisation = config["upper_optimisation"]
    lower_level_train = create_train(
        env,
        env_params,
        config["lower_optimisation"],
        return_transition=False,
    )
    get_trajectory_batch = create_trajectory_batch_sample(
        config,
        env,
        env_params,
    )

    def estimate_discounted_value(
        key: jax.random.PRNGKey, env_params_estimate: EnvParams
    ) -> Tuple[jax.Array, jax.Array]:
        key, _rng = jax.random.split(key)
        train_outputs = lower_level_train(
            _rng,
            env_params_estimate,
            None,
        )
        key, _rng = jax.random.split(key)
        traj_batch = get_trajectory_batch(
            _rng,
            train_outputs["runner_state"][0],
            env_params_estimate,
            0.0,  # Epsilon greedy parameter
        )
        _, discounted_social_welfare = calculate_discounted_rewards(
            env_params_estimate,
            env.social_welfare,
            traj_batch,
            config_upper_optimisation["discount_factor"],
            initial_value=jnp.zeros((config_lower_training["num_envs"],)),
        )  # Shape: (n_steps, num_envs)
        V_UL = jnp.nanmean(
            jnp.where(traj_batch.state.time == 0, discounted_social_welfare, jnp.nan)
        )  # Shape: (num_envs,)
        discounting_arr = jnp.power(
            config["lower_optimisation"]["discount_factor"],
            traj_batch.state.time,
        )
        num_episodes = jnp.sum(traj_batch.done)  # Shape: (num_envs,)
        V_LL = jnp.sum(discounting_arr * traj_batch.reward) / num_episodes
        return V_UL, V_LL

    def update_step(carry, t):
        (
            rng_carry,
            env_params_train_carry,
            upper_level_train_states_carry,
            consumption_preferences_fixed,
        ) = carry

        # Realize Xi
        xi_cardinality = consumption_preferences_fixed.shape[0]
        rng_carry, _rng = jax.random.split(rng_carry)
        xi_idx = jax.random.randint(_rng, (), minval=0, maxval=xi_cardinality)
        env_params_fixed_xi = env_params_train_carry.replace(
            reward_params=env_params_train_carry.reward_params.replace(
                consumption_preferences=consumption_preferences_fixed[xi_idx]
            )
        )

        # Estimate Value for Xi
        rng_carry, _rng = jax.random.split(rng_carry)
        V_UL, V_LL = estimate_discounted_value(_rng, env_params_fixed_xi)

        # Sample Z and u
        rng_carry, _rng = jax.random.split(rng_carry)
        Z = random_normal_perturbations(_rng, upper_level_train_states_carry)
        u = config_upper_optimisation["zero_order_perturbation_constant"] / t

        # Estimate Value for perturbed parameters
        upper_level_train_states_perturbed = {
            key: ts.replace(
                params={
                    "params": {"weights": ts.params["params"]["weights"] + u * Z[key]}
                }
            )
            for key, ts in upper_level_train_states_carry.items()
        }
        env_params_tmp = update_tax_params(
            env_params_fixed_xi, upper_level_train_states_perturbed
        )
        rng_carry, _rng = jax.random.split(rng_carry)
        V_UL_perturbed, V_LL_perturbed = estimate_discounted_value(_rng, env_params_tmp)

        # Update parameters
        grad = {
            key: {"params": {"weights": -(V_UL_perturbed - V_UL) * z / u}}
            for key, z in Z.items()
        }
        upper_level_train_states_carry = {
            key: ts.apply_gradients(
                grads=flax.core.FrozenDict(grad[key])
                if jax.__version__ == "0.4.10"
                else grad[key],
            )
            for key, ts in upper_level_train_states_carry.items()
        }

        # Update the environment parameters
        env_params_train_carry = update_tax_params(
            env_params_train_carry, upper_level_train_states_carry
        )

        metrics = {
            "xi_idx": xi_idx,
            "V_UL": V_UL,
            "V_UL_perturbed": V_UL_perturbed,
            "V_LL": V_LL,
            "V_LL_perturbed": V_LL_perturbed,
            "vat": env_params_train_carry.reward_params.consumption_tax_rate,
            "income_tax": env_params_train_carry.transition_params.income_tax_rate,
            "vat_grad": grad["vat"]["params"]["weights"],
            "income_tax_grad": grad["income_tax"]["params"]["weights"],
        }
        return (
            rng_carry,
            env_params_train_carry,
            upper_level_train_states_carry,
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
        update_dict, is_leaf=lambda x: isinstance(x, jnp.ndarray)
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
    basic_env, env_params = setup_environment(config_init)
    print("Env params: ", env_params)

    consumption_preferences_fixed = jnp.array(
        config["environment"]["params"]["reward_params"]["consumption_preferences"]
    )

    def run_experiment(
        key: jax.random.PRNGKey,
        config_update: Dict[str, Any],
    ) -> Tuple[
        Tuple[jnp.ndarray, EnvParams, Dict[str, TrainState], jnp.ndarray],
        Tuple[jnp.ndarray],
    ]:
        config_exp = deepcopy(config_init)
        config_exp = update_dictionary(config_exp, config_update)
        env_params_exp = update_nested_pytree(
            env_params, config_exp["environment"]["params"]
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
        env_params_exp = update_tax_params(env_params_exp, upper_level_train_states)

        # TRAINING
        update_step = create_update_step(basic_env, env_params_exp, config_exp)
        n_iter = config_exp["upper_optimisation"]["num_outer_iter"]
        return jax.lax.scan(
            update_step,
            (
                key,
                env_params_exp,
                upper_level_train_states,
                consumption_preferences_fixed,
            ),
            jnp.arange(1, n_iter + 1),
            n_iter,
        )

    start_time = time.time()
    if len(update_dict) > 0:
        run_experiment_vmap = jax.vmap(
            jax.vmap(
                run_experiment, in_axes=(None, jax.tree_map(lambda x: 0, update_dict))
            ),
            in_axes=(0, None),
        )
        carry_out, output_metrics = jax.block_until_ready(
            jax.jit(run_experiment_vmap)(
                jax.random.split(rng, config_init["num_seeds"]), update_dict
            )
        )
    else:
        run_experiment_vmap = jax.vmap(run_experiment, in_axes=(0, None))
        carry_out, output_metrics = run_experiment_vmap(
            jax.random.split(rng, config_init["num_seeds"]), update_dict
        )
    run_time = time.time() - start_time
    print(
        f"Experiment runtime: {(run_time) / 60:.2f} minutes and {(run_time) % 60:.2f} seconds"
    )
    _, env_params, upper_level_train_states, _ = carry_out

    # Save results
    with open(os.path.join(experiment_dir, "metrics_zero_order.pkl"), "wb") as f:
        pickle.dump(output_metrics, f)
    with open(os.path.join(experiment_dir, "update_dict_zero_order.pkl"), "wb") as f:
        pickle.dump(update_dict, f)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    orbax_checkpointer.save(
        os.path.join(
            os.path.abspath(experiment_dir), "checkpoint_incentive_zero_order"
        ),
        upper_level_train_states,
        save_args=orbax_utils.save_args_from_target(upper_level_train_states),
        force=True,
    )

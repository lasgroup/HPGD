import flax.struct
import jax
import jax.numpy as jnp

import flashbax as fbx
import optax
import distrax
import gymnax

from typing import Dict, Sequence, Union
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
from src.algorithms.Regularized_DQN import create_train_state, get_actions_from_q_values


import collections.abc


def update_dictionary(dictionary, update):
    for k, v in update.items():
        if isinstance(v, collections.abc.Mapping):
            dictionary[k] = update_dictionary(dictionary.get(k, {}), v)
        else:
            dictionary[k] = v
    return dictionary


@flax.struct.dataclass
class Transition:
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    state: jax.Array = None


def get_random_action(
    key: jax.random.PRNGKey,
    num_actions: int,
    num_envs: int,
    action_space_shape: Sequence[Union[int, jax.Array]],
):
    """
    Get random action from the action space.
    """
    random_action = jax.random.randint(
        key, minval=0, maxval=num_actions, shape=(num_envs,)
    )
    random_action = jnp.unravel_index(random_action, action_space_shape)
    random_action = jnp.stack(random_action, -1)  # Shape: (num_envs, num_actions)
    return random_action


def create_train(
    basic_env: gymnax.environments.environment.Environment,
    env_params_create: gymnax.environments.environment.EnvParams,
    config_regularized_DQN: Dict,
    return_transition: bool = False,
):
    """
    Create the training function for the Regularized DQN algorithm.
    The algorithm follows: "A Theory of Regularized Markov Decision Processes, Geist et al., 2019"
    DQN framework is adapted from: https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/dqn.py
    """
    env = FlattenObservationWrapper(basic_env)
    env = LogWrapper(env)

    vmap_reset = lambda n_envs: lambda rng, env_params_reset: jax.vmap(
        env.reset, in_axes=(0, None)
    )(jax.random.split(rng, n_envs), env_params_reset)
    vmap_step = lambda n_envs: lambda rng, env_state, action, env_params_step: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params_step)

    action_space = basic_env.action_space(env_params_create)
    action_space_shape = [s.n for s in action_space.spaces]
    num_actions = jnp.prod(jnp.array(action_space_shape))

    def train(
        rng: jax.random.PRNGKey,
        env_params_train: gymnax.environments.environment.EnvParams,
        config_update: Dict = None,
    ):
        # UPDATE CONFIG
        if config_update is not None:
            config_regularized_DQN_train = update_dictionary(
                config_regularized_DQN, config_update
            )
        else:
            config_regularized_DQN_train = config_regularized_DQN
        config_lower_training = config_regularized_DQN_train["training"]
        config_epsilon_greedy = config_lower_training["epsilon_greedy"]
        reg_lambda = config_regularized_DQN_train["reg_lambda"]

        # CREATE AGENT
        rng, _rng = jax.random.split(rng)
        agent_train_state = create_train_state(
            _rng,
            config_regularized_DQN_train["network_params"],
            basic_env,
            env_params_train,
        )
        init_train_state_params_target = agent_train_state.params

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        init_obs, env_state_init = vmap_reset(config_lower_training["num_envs"])(
            _rng, env_params_train
        )

        # INIT BUFFER
        buffer = fbx.make_flat_buffer(
            max_length=config_lower_training["buffer_size"],
            min_length=config_lower_training["batch_size"],
            sample_batch_size=config_lower_training["batch_size"],
            add_sequences=False,
            add_batch_size=config_lower_training["num_envs"],
        )
        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )
        rng_dummy = jax.random.PRNGKey(0)  # use a dummy rng here
        rng_dummy_sample, rng_dummy_reset, rng_dummy_step = jax.random.split(
            rng_dummy, 3
        )
        _action = jnp.array(action_space.sample(rng_dummy_sample))
        _, _env_state = env.reset(rng_dummy_reset, env_params_train)
        _obs, _, _reward, _done, _ = env.step(
            rng_dummy_step, _env_state, _action, env_params_train
        )
        _timestep = Transition(obs=_obs, action=_action, reward=_reward, done=_done)
        buffer_state_init = buffer.init(_timestep)

        # TRAINING LOOP
        def _update_step(runner_state, timestep_cnt):
            (
                train_state,
                train_state_params_target,
                buffer_state,
                env_state,
                last_obs,
                rng_step,
            ) = runner_state

            # STEP THE ENV
            rng_step, rng_r, rng_a, rng_s = jax.random.split(rng_step, 4)
            q_values = train_state.apply_fn(
                train_state.params, last_obs
            )  # Shape: (num_envs, num_actions) or list of (num_envs, action_dim) for action dimensions
            action, entropy = get_actions_from_q_values(
                rng_a, q_values, reg_lambda, action_space_shape=action_space_shape
            )
            random_action = get_random_action(
                rng_r,
                num_actions,
                config_lower_training["num_envs"],
                action_space_shape,
            )
            eps = (
                config_epsilon_greedy["start"]
                - jnp.clip(timestep_cnt, 0, config_epsilon_greedy["anneal_time"])
                * (config_epsilon_greedy["start"] - config_epsilon_greedy["end"])
                / config_epsilon_greedy["anneal_time"]
            )
            action = jax.lax.select(
                jax.random.bernoulli(rng_r, eps),
                random_action,
                action,
            )
            obs, env_state, reward, done, info = vmap_step(
                config_lower_training["num_envs"]
            )(rng_s, env_state, action, env_params_train)

            # BUFFER UPDATE
            transition = Transition(
                obs=last_obs, action=action, reward=reward, done=done
            )
            buffer_state = buffer.add(buffer_state, transition)

            # NETWORKS UPDATE
            def _learn_phase(train_state_learn, _rng):
                learn_batch = buffer.sample(buffer_state, _rng).experience
                q_next_target = train_state_learn.apply_fn(
                    train_state_params_target, learn_batch.second.obs
                )  # Shape: (batch_size, num_actions) or Sequence of (batch_size, action_dim)
                q_next_target = jax.tree_map(
                    lambda x: reg_lambda
                    * jax.scipy.special.logsumexp(x / reg_lambda, axis=-1),
                    q_next_target,
                )  # Shape: (batch_size,) or Sequence of (batch_size, )
                target = jax.tree_map(
                    lambda x: learn_batch.first.reward
                    + (1 - learn_batch.first.done)
                    * config_regularized_DQN_train["discount_factor"]
                    * x,
                    q_next_target
                )  # Shape: (batch_size,) or Sequence of (batch_size, )

                def _loss_fn(params):
                    q_vals = train_state_learn.apply_fn(
                        params, learn_batch.first.obs
                    )  # (batch_size, action_space_shape_prod) or Sequence of (batch_size, action_dim)
                    if isinstance(q_vals, jax.Array):
                        chosen_action = jax.vmap(
                            lambda x: jnp.ravel_multi_index(
                                x, action_space_shape, mode="clip"
                            )
                        )(
                            learn_batch.first.action
                        )  # (batch_size,)
                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(chosen_action, axis=-1),
                            axis=-1,
                        ).squeeze(
                            axis=-1
                        )  # (batch_size,)
                        return jnp.mean((chosen_action_qvals - target) ** 2)
                    else:
                        chosen_action_qvals = jax.tree_map(
                            lambda x, y: (
                                jnp.take_along_axis(
                                    x, jnp.expand_dims(y, axis=-1), axis=-1
                                ).squeeze(axis=-1)
                            ),
                            q_vals,
                            list(learn_batch.first.action.T),
                        )  # Sequence of (batch_size,)
                        return jnp.sum(
                            jnp.stack(
                                jax.tree_map(
                                    lambda x, y: jnp.mean((x - y) ** 2), chosen_action_qvals, target
                                )
                            )
                        )

                loss, grads = jax.value_and_grad(_loss_fn)(train_state_learn.params)
                train_state_learn = train_state_learn.apply_gradients(grads=grads)
                return train_state_learn, loss

            is_learn_time = (
                (buffer.can_sample(buffer_state))  # enough experience in buffer
                & (  # pure exploration phase ended
                    timestep_cnt * config_lower_training["num_envs"]
                    > config_lower_training["learning_starts"]
                )
                & (  # training interval
                    timestep_cnt
                    * config_lower_training["num_envs"]
                    % config_lower_training["training_interval"]
                    == 0
                )
            )
            rng_step, rng_learn = jax.random.split(rng_step)
            train_state, loss = jax.lax.cond(
                is_learn_time,
                lambda _train_state, rng_key: _learn_phase(_train_state, rng_key),
                lambda _train_state, rng_key: (_train_state, jnp.nan),  # do nothing
                train_state,
                rng_learn,
            )

            # update target network
            train_state_params_target = jax.lax.cond(
                timestep_cnt
                * config_lower_training["num_envs"]
                % config_lower_training["target_update_interval"]
                == 0,
                lambda: optax.incremental_update(
                    train_state.params,
                    train_state_params_target,
                    config_lower_training["target_update_learning_rate"],
                ),
                lambda: train_state_params_target,
            )

            metrics = {
                # "updates": train_state.n_updates,
                "loss": jnp.nanmean(loss).astype(jnp.float16),
                "returns": info["returned_episode_returns"].mean().astype(jnp.float16),
                "entropy": jnp.mean(entropy, 0).astype(jnp.float16),
                "obs": jnp.mean(obs).astype(jnp.float16),
                "action": jnp.mean(action, 0).astype(jnp.float16),
                # "social_welfare": info["social_welfare"].mean(),
            }
            if return_transition:
                metrics["transition"] = transition

            runner_state = (
                train_state,
                train_state_params_target,
                buffer_state,
                env_state,
                obs,
                rng_step,
            )

            return runner_state, metrics

        # train
        rng, _rng = jax.random.split(rng)
        initial_runner_state = (
            agent_train_state,
            init_train_state_params_target,
            buffer_state_init,
            env_state_init,
            init_obs,
            _rng,
        )

        runner_state, metrics = jax.lax.scan(
            _update_step,
            initial_runner_state,
            jnp.arange(1, config_lower_training["num_steps"] + 1),
            config_lower_training["num_steps"],
        )
        return {
            "runner_state": runner_state,
            "metrics": metrics,
            "init_train_state_params_target": init_train_state_params_target,
        }

    return train

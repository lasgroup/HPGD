import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional, Iterable, Dict, Callable, Union

from flax import struct
import distrax

from src.environments.utils import default_reward_function
from src.environments.utils import (
    FOUR_ROOMS_DEFAULT_MAP,
    FOUR_ROOMS_DEFAULT_CORRIDOR_COORDS,
    string_to_bool_map,
    map_project,
    sample_array,
    get_coordinates,
)


@struct.dataclass
class EnvState:
    pos: jax.Array
    goal: jax.Array
    time: int


@struct.dataclass
class ConfFourRoomsParams:
    fail_prob: float = 1.0 / 3.0
    max_steps_in_episode: int = 100
    resample_goal_logits: jnp.ndarray = jnp.nan
    state_initialization_params: Union[dict, jnp.ndarray] = jnp.nan
    incentive_params: Union[dict, jnp.ndarray] = jnp.nan
    reward_function_params: Union[dict, jnp.ndarray] = jnp.nan
    corridor_door_closed_prob: jnp.ndarray = (0.0, 0.0, 0.0, 0.0)


class ConfigurableFourRooms(environment.Environment):
    """
    JAX Compatible version of Four Rooms environment (Sutton et al., 1999).
    Source: Comparable to https://github.com/howardh/gym-fourrooms
    Since gymnax automatically resets env at done, we abstract different resets

    Based on the gymnax implementation of the FourRooms environment.
    Source: https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/misc/rooms.py
    """

    def __init__(
        self,
        use_visual_obs: bool = False,
        available_goals: Iterable[Iterable[int]] = ((8, 9),),
        available_init_pos: Iterable[Iterable[int]] = None,
        state_initialization_distribution: Callable[
            [Union[dict, jnp.ndarray]], distrax.Distribution
        ] = lambda x: distrax.Categorical(logits=x),
        reward_function: Union[
            Callable[[EnvState, jnp.ndarray, Union[dict, jnp.ndarray]], jnp.ndarray],
            str,
        ] = "default",
        incentive_function: Callable[
            [EnvState, jnp.ndarray, Union[dict, jnp.ndarray]], jnp.ndarray
        ] = lambda s, a, params: jnp.array(0.0),
    ):
        """
        Initialize the FourRooms environment.
        :param use_visual_obs:
        :param available_goals:
        :param available_init_pos:
        :param state_initialization_distribution: Distribution over the initial states
            s_0 ~ p(params) where params is ConfFourRoomsParams.state_initialization_params
        :param reward_function:  Reward function returning the reward r(s,a)
            the third argument is the reward_function_parameters
        :param incentive_function: Incentive function returning the incentive i(s,a)
            the third argument is the incentive_function_parameters
        """
        super().__init__()
        self.env_map = string_to_bool_map(FOUR_ROOMS_DEFAULT_MAP)
        self.occupied_map = 1 - self.env_map
        self.coords = get_coordinates(FOUR_ROOMS_DEFAULT_MAP)
        self.directions = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
        self.corridor_coords = jnp.array(FOUR_ROOMS_DEFAULT_CORRIDOR_COORDS)

        # available goals and initial positions, if None use all available positions
        self.available_goals = (
            jnp.array(available_goals) if available_goals else self.coords
        )
        self.available_init_pos = (
            jnp.array(available_init_pos) if available_init_pos else self.coords
        )
        self.state_initialization_distribution = state_initialization_distribution
        # Reward function
        if reward_function == "default" or reward_function is None:
            self.reward_function = default_reward_function
        elif isinstance(reward_function, Callable):
            self.reward_function = reward_function
        else:
            raise ValueError("Only 'default' or a callable reward function is allowed.")
        # Incentive function
        self.incentive_function = incentive_function

        # Whether to use 3D visual observation
        # Channel ID 0 - Wall (1) or not occupied (0)
        # Channel ID 1 - Agent location in maze
        self.use_visual_obs = use_visual_obs

    @property
    def name(self) -> str:
        """Environment name."""
        return "FourRooms-misc"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 4

    @property
    def default_params(self) -> ConfFourRoomsParams:
        return ConfFourRoomsParams(
            resample_goal_logits=jnp.ones(len(self.available_goals)),
        )

    def action_space(
        self, params: Optional[ConfFourRoomsParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(4)

    def observation_space(self, params: ConfFourRoomsParams) -> spaces.Box:
        """Observation space of the environment."""
        if self.use_visual_obs:
            return spaces.Box(0, 1, (13, 13, 2), jnp.float32)
        else:
            return spaces.Box(
                jnp.min(self.coords), jnp.max(self.coords), (4,), jnp.float32
            )

    def state_space(self, params: ConfFourRoomsParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "pos": spaces.Box(
                    jnp.min(self.coords),
                    jnp.max(self.coords),
                    (2,),
                    jnp.float32,
                ),
                "goal": spaces.Box(
                    jnp.min(self.coords),
                    jnp.max(self.coords),
                    (2,),
                    jnp.float32,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

    def _transition_kernel(
        self, pos: jnp.ndarray, action_idx: jnp.ndarray
    ) -> jnp.ndarray:
        """Transition kernel for a given position and action."""
        new_pos = pos + self.directions[action_idx]
        return map_project(self.env_map, pos, new_pos)

    def get_reward(
            self,
            state: EnvState,
            action: int,
            params: ConfFourRoomsParams
    ) -> float:
        reward = self.reward_function(state, action, params.reward_function_params)
        incentive = self.incentive_function(state, action, params.incentive_params)
        return reward + incentive

    def step_env(
        self,
        key: jax.random.PRNGKey,
        state: EnvState,
        action: int,
        params: ConfFourRoomsParams,
    ) -> Tuple[jax.Array, EnvState, float, bool, Dict[str, float]]:
        """Perform single timestep state transition."""
        # Update counter
        state = state.replace(time=state.time + 1)
        key_random, key_action, key_corridor = jax.random.split(key, 3)
        # Sample whether to choose a random action
        choose_random = jax.random.uniform(key_random, ()) < params.fail_prob * 4 / 3
        action = jax.lax.select(
            choose_random, self.action_space(params).sample(key_action), action
        )
        # Calculate reward
        reward = self.get_reward(state, action, params)
        # Check if terminal
        done = self.is_terminal(state, params)

        # Transition
        new_pos = self._transition_kernel(state.pos, action)
        # Adjust if new position is in a corridor and the door is closed
        in_corridor_mask = jnp.all(
            self.corridor_coords == new_pos, 1
        )  # Shape: (num_corridors,)
        corridor_closed = jax.random.uniform(key_corridor, ()) < jnp.sum(
            jnp.where(in_corridor_mask, jnp.array(params.corridor_door_closed_prob), 0.0)
        )
        new_pos = jax.lax.select(
            corridor_closed, state.pos, new_pos
        )  # If door is closed, stay at the same position
        # If terminal, stay at the same position
        new_pos = jax.lax.select(done, state.pos, new_pos)
        state = state.replace(pos=new_pos)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: jax.random.PRNGKey, params: ConfFourRoomsParams
    ) -> Tuple[jax.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        # Reset both the agents position and the goal location
        rng_goal, rng_pos = jax.random.split(key, 2)
        goal, _, _ = sample_array(
            rng_goal, self.available_goals, params.resample_goal_logits
        )
        state_distribution = self.state_initialization_distribution(
            params.state_initialization_params
        )
        pos_idx = state_distribution.sample(seed=rng_pos)
        pos = self.available_init_pos[pos_idx]
        state = EnvState(pos, goal, 0)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> jax.Array:
        """Return observation from raw state trafo."""
        if not self.use_visual_obs:
            return jnp.array(
                [
                    state.pos[0],
                    state.pos[1],
                    state.goal[0],
                    state.goal[1],
                ]
            )
        else:
            agent_map = jnp.zeros(self.occupied_map.shape)
            agent_map = agent_map.at[state.pos[1], state.pos[0]].set(1)
            obs_array = jnp.stack([self.occupied_map, agent_map], axis=2)
            return obs_array

    def is_terminal(self, state: EnvState, params: ConfFourRoomsParams) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done_steps = state.time >= params.max_steps_in_episode
        # Check if agent has found the goal
        done_goal = jnp.logical_and(
            state.pos[0] == state.goal[0],
            state.pos[1] == state.goal[1],
        )
        done = jnp.logical_or(done_goal, done_steps)
        return done

    @property
    def terminal_states(self) -> jnp.ndarray:
        """
        Calculate goal state rewards for the environment.
        Returns a jnp.Array with shape (len(env.available_goals), n_states)
        """
        return jnp.all(self.coords[None, :, :] == self.available_goals[:, None, :], 2)

    def render(
        self,
        state: EnvState,
        params: ConfFourRoomsParams,
        ax: Optional = None,
        annotate_positions: bool = True,
    ):
        """Small utility for plotting the agent's state."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None
        ax.imshow(self.occupied_map, cmap="Greys")
        if annotate_positions:
            ax.annotate(
                "A",
                fontsize=20,
                xy=(state.pos[1], state.pos[0]),
                xycoords="data",
                xytext=(state.pos[1] - 0.3, state.pos[0] + 0.25),
            )
            ax.annotate(
                "G",
                fontsize=20,
                xy=(state.goal[1], state.goal[0]),
                xycoords="data",
                xytext=(state.goal[1] - 0.3, state.goal[0] + 0.25),
            )
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax

    def get_transition_probability_matrix(
        self, params: ConfFourRoomsParams
    ) -> jnp.ndarray:
        """
        Calculate transition probabilities for the environment.
        Returns a jnp.Array with shape (len(env.available_goals), n_states, n_actions, n_states)
        """
        # Calculate the probability of being stopped by a door at state s_{t+1}
        door_state_mask = jnp.all(
            self.coords[:, None, :] == self.corridor_coords[None, :, :], 2
        )  # Shape: (n_states, num_corridors)
        door_state_prob = jnp.sum(
            door_state_mask * jnp.array(params.corridor_door_closed_prob), 1
        )  # P[hit door | s_{t+1}], Shape: (n_states,)
        adjusted_fail_prob = 4 * params.fail_prob / 3

        def state_action_transition_probability(
            state_idx: jnp.ndarray, action_idx: jnp.ndarray
        ) -> jnp.ndarray:
            """Calculate the transition probability for a given state and action pair. Considering the doors."""
            current_pos = self.coords[state_idx]
            new_pos = self._transition_kernel(current_pos, action_idx)
            new_pos_mask = jnp.all(
                self.coords == new_pos[None, :], 1
            )  # Shape: (n_states,)
            hit_door_prob = jnp.sum(door_state_prob * new_pos_mask)
            transition_prob = (1 - hit_door_prob) * new_pos_mask.astype(jnp.float32)
            return transition_prob.at[state_idx].set(
                transition_prob[state_idx] + hit_door_prob
            )  # Shape: (n_states,)

        def state_noisy_action_transition_probability(
            state_idx: jnp.ndarray, action_idx: jnp.ndarray
        ) -> jnp.ndarray:
            """Calculate the transition probability for a given state and action pair. Considering the action noise."""
            state_action_probs = jax.vmap(
                state_action_transition_probability, in_axes=(None, 0), out_axes=1
            )(
                state_idx, jnp.arange(self.num_actions)
            )  # Shape: (n_states, n_actions)
            action_probs = jnp.where(
                jnp.arange(self.num_actions) == action_idx,
                1 - adjusted_fail_prob,
                adjusted_fail_prob / (self.num_actions - 1),
            )
            return jnp.sum(state_action_probs * action_probs[None, :], 1)

        state_noisy_action_transition_probability_vmap = jax.vmap(
            jax.vmap(state_noisy_action_transition_probability, in_axes=(None, 0)),
            in_axes=(0, None),
        )
        P = state_noisy_action_transition_probability_vmap(
            jnp.arange(self.coords.shape[0]), jnp.arange(self.num_actions)
        )  # Shape: (n_states, n_actions, n_states)

        # Mask goal states
        def mask_goal(goal):
            mask = jnp.all(self.coords == goal[None, :], 1)
            P_masked = jnp.where(~mask[:, None, None], P, 0)
            P_masked = P_masked + jnp.where(
                mask[:, None, None] * mask[None, None, :], 1, 0
            )
            return P_masked

        mask_goal_v = jax.vmap(mask_goal)
        return mask_goal_v(
            self.available_goals
        )  # Shape: (len(env.available_goals), n_states, n_actions, n_states)


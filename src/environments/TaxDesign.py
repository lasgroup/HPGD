import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional, Dict, Callable, Union

from flax import struct


@struct.dataclass
class EnvState:
    assets: jnp.ndarray
    time: int


@struct.dataclass
class RewardParams:
    accumulated_asset_utility_scale: float = 0.1
    work_disutility: float = 0.1
    prices: jnp.ndarray = (1.0, 1.0, 1.0)
    consumption_preferences: jnp.ndarray = (0.5, 0.5, 0.5)  # Shape: (n_goods)
    consumption_tax_rate: jnp.ndarray = (0.1, 0.1, 0.1)  # Shape: (n_goods)
    social_welfare_tax_utility: float = 1.0
    social_welfare_consumption_scale: float = 1.0
    social_welfare_asset_utility_scale: float = 2.0


@struct.dataclass
class TransitionParams:
    income_tax_rate: float = 0.1
    wage: float = 1.0
    asset_range: Tuple[float, float] = (-100.0, 100.0)
    transition_std: float = 0.1


@struct.dataclass
class EnvParams(environment.EnvParams):
    max_steps_in_episode: int = 100
    reward_params: RewardParams = RewardParams()
    transition_params: TransitionParams = TransitionParams()


class TaxDesign(environment.Environment):
    def __init__(
        self,
        accumulated_asset_utility: Callable = lambda x, scale: scale*x,
        n_goods: int = 3,
        max_income_tax: float = 0.4,
        max_consumption_tax: float = 0.4,
        action_discretization: Dict[str, Union[int, float]] = {
            "hours_worked_n": 10,
            "hours_worked_scale": 8 / 9,
            "consumption_n": 5,
            "consumption_scale": 1.25,
        }
    ):
        """
        Action ranges for discretization
        - hours_worked: [0, 8]
        - consumption: [0, 5]
        :param accumulated_asset_utility:
        :param n_goods:
        """
        self.accumulated_asset_utility = accumulated_asset_utility
        self.action_discretization = action_discretization

        self.ActionSpace = spaces.Tuple(
            [spaces.Discrete(self.action_discretization["hours_worked_n"])]
            + [
                spaces.Discrete(self.action_discretization["consumption_n"])
                for _ in range(n_goods)
            ]
        )
        self.max_income_tax = max_income_tax
        self.max_consumption_tax = max_consumption_tax

    @property
    def name(self) -> str:
        """Environment name."""
        return "TaxDesign"

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Space:
        """Action space of the environment."""
        return self.ActionSpace

    def observation_space(self, params: EnvParams) -> spaces.Space:
        """Observation space of the environment, i.e., accumulated assets"""
        asset_range = params.transition_params.asset_range
        return spaces.Box(asset_range[0], asset_range[1], (1,), jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Space:
        """State space of the environment."""
        return spaces.Dict(
            {
                "obs": self.observation_space(params),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check if the state is terminal."""
        return state.time >= params.max_steps_in_episode

    def _cobb_douglas_utility(
        self,
            consumption: jax.Array,
            preferences: jax.Array,
            prices: jax.Array,
            tax_rate: jax.Array,
    ) -> jax.Array:
        """Compute utility from consumption."""
        return jnp.prod(
            jnp.power(consumption / (prices * (1 + tax_rate) + 1e-16), preferences)
        )

    def reward(
        self,
        state: EnvState,
        action: jax.Array,
        params: RewardParams,
    ) -> jax.Array:
        """Compute reward from state, action."""
        action_continuous = self.action_discretized_to_continuous(action)
        working_hours = action_continuous[0]
        consumption = action_continuous[1:]
        asset_utility = self.accumulated_asset_utility(state.assets, params.accumulated_asset_utility_scale)
        work_disutility = params.work_disutility * working_hours**2
        consumption_utility = self._cobb_douglas_utility(
            consumption,
            params.consumption_preferences,
            params.prices,
            params.consumption_tax_rate,
        )
        raw_reward = asset_utility - work_disutility + consumption_utility
        return raw_reward

    def transition(
        self, state: EnvState, action: jax.Array, params_transition: TransitionParams
    ) -> EnvState:
        """Compute next state from state, action."""
        action_continuous = self.action_discretized_to_continuous(action)
        working_hours = action_continuous[0]
        consumption = jnp.sum(action_continuous[1:])
        new_assets = (
            state.assets
            + (1 - params_transition.income_tax_rate)
            * working_hours
            * params_transition.wage
            - consumption
        )

        # Clip assets to the range
        range_min, range_max = params_transition.asset_range
        new_assets = jnp.clip(
            new_assets,
            range_min,
            range_max,
        )

        return EnvState(assets=new_assets, time=state.time + 1)

    def add_transition_noise(
        self, key: jax.random.PRNGKey, new_state: EnvState, params: TransitionParams
    ) -> EnvState:
        """
        Add truncated normal noise to the transition.
        """
        std = params.transition_std
        range_min, range_max = params.asset_range
        new_assets = new_state.assets
        noise = jax.random.truncated_normal(
            key=key,
            lower=(range_min - new_assets) / std,
            upper=(range_max - new_assets) / std,
            shape=new_assets.shape,
        )
        new_assets = new_assets + std * noise
        new_state = new_state.replace(assets=new_assets)
        return new_state

    def action_discretized_to_continuous(self, action: jax.Array) -> jax.Array:
        """Scale discrete action to continuous action."""
        hours_worked = action[0] * self.action_discretization["hours_worked_scale"]
        consumption = action[1:] * self.action_discretization["consumption_scale"]
        return jnp.concatenate([jnp.atleast_1d(hours_worked), consumption])

    def step_env(
        self,
        key: jax.random.PRNGKey,
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
    ) -> Tuple[jax.Array, EnvState, jax.Array, bool, Dict[str, float]]:
        """Execute a step of the environment."""
        # Compute reward
        reward = self.reward(
            state,
            action,
            params.reward_params,
        )

        # Update state
        new_state = self.transition(state, action, params.transition_params)
        new_state = jax.tree_map(
            lambda x, y: jax.lax.select(
                params.transition_params.transition_std > 0, x, y
            ),
            self.add_transition_noise(key, new_state, params.transition_params),
            new_state
        )

        # Social Welfare
        social_welfare = self.social_welfare(
            state,
            action,
            params,
        )

        done = self.is_terminal(new_state, params)
        return (
            lax.stop_gradient(self.get_obs(new_state)),
            lax.stop_gradient(new_state),
            reward,
            done,
            {
                "discount": self.discount(new_state, params),
                "social_welfare": social_welfare,
            },
        )

    def reset_env(
        self, key: jax.random.PRNGKey, params: EnvParams
    ) -> Tuple[jax.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        # Reset both the agents position and the goal location
        state = EnvState(
            assets=jnp.sqrt(2) * jax.random.normal(key),
            time=0,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> jax.Array:
        """Return observation from raw state."""
        return state.assets

    def social_welfare(
        self,
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
    ) -> jax.Array:
        """
        Compute social welfare from state.
        """
        reward_params = params.reward_params
        transition_params = params.transition_params
        action_continuous = self.action_discretized_to_continuous(action)
        working_hours = action_continuous[0]
        consumption = action_continuous[1:]
        max_consumption = (
                self.action_discretization["consumption_scale"]
                * (self.action_discretization["consumption_n"] - 1)
                * jnp.ones_like(reward_params.prices)
        )

        accumulated_asset = self.accumulated_asset_utility(state.assets, reward_params.social_welfare_asset_utility_scale)

        def f_consumption_utility(consumption_arr, vat_tax_rate):
            """
            Compute utility from consumption.
            Range: [0, 1]
            """
            return jnp.mean(consumption_arr / (1 + vat_tax_rate + 1e-16)) / jnp.max(max_consumption)

        def f_tax_utility(consumption_arr, vat_tax_rate, wage, income_tax_rate, hours_worked):
            return jnp.log(
                1  # Shift by one to avoid negative values
                + jnp.sum(vat_tax_rate * consumption_arr / (1+vat_tax_rate + 1e-16))
                + income_tax_rate * wage * hours_worked
            )

        consumption_utility = (
            reward_params.social_welfare_consumption_scale
            * f_consumption_utility(consumption, reward_params.consumption_tax_rate)
        )

        tax_utility = reward_params.social_welfare_tax_utility * f_tax_utility(
            consumption,
            reward_params.consumption_tax_rate,
            transition_params.wage,
            transition_params.income_tax_rate,
            working_hours
        )
        raw_reward = accumulated_asset + consumption_utility + tax_utility
        return raw_reward

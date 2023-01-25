import numpy as np
from gym import Env

from src.ours.env.component.visualizer import (
    TrajectoryHeatVisualizer,
    PositionVisualizer,
)
from src.ours.env.component.field import Field
from src.ours.env.util.space import (
    DiscreteSpacesGenerator,
    ContSpacesGenerator,
)
from src.ours.env.util.action import DiscreteActionConverter, ContActionConverter
from src.ours.env.util.renderer import HumanPointEnvRenderer, RgbPointEnvRenderer
from src.ours.env.util.time import EpisodeLengthTimer


class PointNav(Env):
    def __init__(self, n_targets=2, shift_x=0, shift_y=0, random_spawn_agent=False):
        super().__init__()

        self._side_length = 200

        self._board_shape = self._side_length, self._side_length
        self._field = Field(n_targets, (shift_x, shift_y), random_spawn_agent)
        self._position_visualizer = PositionVisualizer(self._field)
        self._trajectory_heat_visualizer = TrajectoryHeatVisualizer(self._field)

        self._episode_timer = EpisodeLengthTimer(1000)

    @property
    def env_config(self) -> dict[str, int]:
        return self._field.config

    def _draw_elements_on_canvas(self) -> None:
        self._position_visualizer.visualize()
        self._trajectory_heat_visualizer.visualize()

    def reset(self) -> np.ndarray:
        self._field.reset()

        self._draw_elements_on_canvas()

        self._episode_timer.reset()

        return self._get_obs()

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action_converted = self._get_action_converted(action)
        reward, has_visited_all_targets = self._field.step(action_converted)

        self._draw_elements_on_canvas()

        obs = self._get_obs()

        has_elapsed = self._episode_timer.advance()

        done = has_visited_all_targets or has_elapsed
        return obs, reward, done, {}

    def _get_action_converted(self, action: int | np.ndarray) -> tuple[int, int]:
        pass

    def _get_obs(self) -> np.ndarray:
        field_obs = self._field.get_pos_agent_target()
        return field_obs

    def render(self, mode="human") -> None:
        assert mode in [
            "human",
            "rgb_array",
        ], 'Invalid mode, must be either "human" or "rgb_array"'

        if mode == "human":
            renderer = HumanPointEnvRenderer(
                self._position_visualizer.colormat,
                self._trajectory_heat_visualizer.colormat,
            )
        else:
            renderer = RgbPointEnvRenderer(self._position_visualizer.colormat)

        renderer.render()

    def close(self) -> None:
        HumanPointEnvRenderer.clean_up()


class DiscretePointNav(PointNav):
    def __init__(self, n_targets=2, shift_x=0, shift_y=0, random_spawn_agent=False):
        super().__init__(
            n_targets,
            shift_x,
            shift_y,
            random_spawn_agent,
        )

        self.observation_space, self.action_space = DiscreteSpacesGenerator(
            self._side_length
        ).get_spaces()

    def _get_action_converted(self, action: int) -> tuple[int, int]:
        return DiscreteActionConverter(action, self.action_space).get_action_converted()


class ContPointNav(PointNav):
    def __init__(self, n_targets=2, shift_x=0, shift_y=0, random_spawn_agent=False):
        super().__init__(
            n_targets,
            shift_x,
            shift_y,
            random_spawn_agent,
        )

        self.observation_space, self.action_space = ContSpacesGenerator(
            self._side_length
        ).get_spaces()

    def _get_action_converted(self, action: np.ndarray) -> tuple[int, int]:
        return ContActionConverter(action, self.action_space).get_action_converted()

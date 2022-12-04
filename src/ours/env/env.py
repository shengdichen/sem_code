from gym import Env

from src.ours.env.component.visualizer import (
    TrajectoryHeatVisualizer,
    PositionVisualizer,
)
from src.ours.env.component.field import Field
from src.ours.env.util.space import SpacesGenerator, ActionConverter
from src.ours.env.util.renderer import PointEnvRendererHuman, PointEnvRendererRgb


class MovePoint(Env):
    def __init__(self, n_targets=2, shift_x=0, shift_y=0, random_spawn_agent=False):
        super(MovePoint, self).__init__()

        self._side_length = 200
        self.observation_space, self.action_space = SpacesGenerator(
            self._side_length
        ).get_spaces()

        self._board_shape = self._side_length, self._side_length
        self._field = Field(n_targets, (shift_x, shift_y), random_spawn_agent)
        self._position_visualizer = PositionVisualizer(self._field)
        self._trajectory_heat_visualizer = TrajectoryHeatVisualizer(self._field)

        self._max_episode_length, self._curr_episode_length = 1000, 0
        self._done = False

    @property
    def env_config(self):
        return self._field.config

    def _draw_elements_on_canvas(self):
        self._position_visualizer.visualize()
        self._trajectory_heat_visualizer.visualize()

    def reset(self):
        self._field.reset()

        self._draw_elements_on_canvas()

        self._curr_episode_length = 0
        self._done = False

        obs = self._get_obs()
        return obs

    def step(self, action: int):
        action_converted = ActionConverter(
            action, self.action_space
        ).get_action_converted()
        reward, self._done = self._field.step(action_converted)

        self._draw_elements_on_canvas()

        obs = self._get_obs()

        self._curr_episode_length += 1
        if self._curr_episode_length == self._max_episode_length:
            self._done = True

        return obs, reward, self._done, {}

    def _get_obs(self):
        field_obs = self._field.get_pos_agent_target()
        return field_obs

    def render(self, mode="human") -> None:
        assert mode in [
            "human",
            "rgb_array",
        ], 'Invalid mode, must be either "human" or "rgb_array"'

        if mode == "human":
            renderer = PointEnvRendererHuman(
                self._position_visualizer.colormat,
                self._trajectory_heat_visualizer.colormat,
            )
        else:
            renderer = PointEnvRendererRgb(self._position_visualizer.colormat)

        renderer.render()

    def close(self):
        PointEnvRendererHuman.clean_up()

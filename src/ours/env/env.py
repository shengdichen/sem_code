from gym import Env

from src.ours.env.canvas import (
    TrajectoryHeatVisualizer,
    AgentTargetsVisualizer,
)
from src.ours.env.component.field import Field
from src.ours.env.space import SpacesGenerator, ActionConverter
from src.ours.env.util import PointEnvRendererHuman, PointEnvRendererRgb


class MovePoint(Env):
    def __init__(self, n_targets=2, shift_x=0, shift_y=0, random_init=False):
        super(MovePoint, self).__init__()

        self._side_length = 200
        self.observation_space, self.action_space = SpacesGenerator(
            self._side_length
        ).get_spaces()

        self._board_shape = self._side_length, self._side_length
        self._field = Field(n_targets, shift_x, shift_y, random_init)
        self._agent_targets_visualizer = AgentTargetsVisualizer(self._board_shape)
        self._trajectory_heat_visualizer = TrajectoryHeatVisualizer(self._board_shape)

        self._max_episode_length, self._curr_episode_length = 1000, 0
        self._done = False

    @property
    def env_config(self):
        return self._field.env_config

    def _draw_elements_on_canvas(self):
        self._agent_targets_visualizer.register(self._field._agent_and_targets)
        self._trajectory_heat_visualizer.register(self._field._agent)

    def reset(self):
        self._field.reset()

        self._draw_elements_on_canvas()

        self._field._curr_tgt_id = 0

        self._curr_episode_length = 0
        self._done = False

        obs = self._get_obs()
        return obs

    def _get_obs(self):
        return self._field._get_obs()

    def step(self, action: int):
        shift = ActionConverter(action, self.action_space).get_shift()
        reward = self._field.step(shift)

        self._update_target()

        self._draw_elements_on_canvas()

        obs = self._get_obs()

        self._curr_episode_length += 1
        if self._curr_episode_length == self._max_episode_length:
            self._done = True

        return obs, reward, self._done, {}

    def _update_target(self):
        self._field._update_target()

    def render(self, mode="human") -> None:
        assert mode in [
            "human",
            "rgb_array",
        ], 'Invalid mode, must be either "human" or "rgb_array"'

        if mode == "human":
            renderer = PointEnvRendererHuman(
                self._agent_targets_visualizer.colormat,
                self._trajectory_heat_visualizer.colormat,
            )
        else:
            renderer = PointEnvRendererRgb(self._agent_targets_visualizer.colormat)

        renderer.render()

    def close(self):
        PointEnvRendererHuman.clean_up()

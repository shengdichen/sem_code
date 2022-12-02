import numpy as np
from gym import Env

from src.ours.env.canvas import (
    TrajectoryHeatVisualizer,
    AgentTargetsVisualizer,
)
from src.ours.env.field import EmptyBoard
from src.ours.env.component.point import PointFactory, NamedPointWithIcon
from src.ours.env.space import SpacesGenerator, ActionConverter
from src.ours.env.util import PointEnvRendererHuman, PointEnvRendererRgb


class Field:
    def __init__(self, n_targets=2, shift_x=0, shift_y=0, random_init=False):
        self._side_length = 200
        self._board_shape = self._side_length, self._side_length
        self._board = EmptyBoard(self._board_shape)

        self._x_range, self._y_range = self._board.movement_ranges
        self._shift_x = shift_x
        self._shift_y = shift_y

        self._random_init = random_init

        self._agent = self._make_agent()

        self._n_tgt = n_targets
        self._curr_tgt_id = 0
        self._targets = self._make_targets()

        self._agent_and_targets = []
        self._agent_and_targets.append(self._agent)
        self._agent_and_targets.extend(self._targets)

    @property
    def env_config(self):
        return {
            "n_targets": self._n_tgt,
            "shift_x": self._shift_x,
            "shift_y": self._shift_y,
        }

    def _make_agent(self) -> NamedPointWithIcon:
        return PointFactory("agent", self._x_range, self._y_range).create_agent()

    def _make_targets(self, make_random_targets=False) -> list[NamedPointWithIcon]:
        targets = []
        for i in range(self._n_tgt):
            tgt = PointFactory(
                "tgt_{}".format(i), self._x_range, self._y_range
            ).create_target()

            # TODO: expand to preferences as random process!
            if make_random_targets:
                tgt_x, tgt_y = self._board.get_target_pos_random()
                tgt.movement.set_position(tgt_x, tgt_y)

            targets.append(tgt)

        return targets

    def reset(self):
        x, y = self._board.get_reset_agent_pos(self._random_init)
        self._agent.movement.set_position(x, y)

        target_positions = self._board.get_two_targets_pos_fixed(
            (self._shift_x, self._shift_y)
        )
        for target, target_pos in zip(self._targets, target_positions):
            target.movement.set_position(target_pos[0], target_pos[1])

    def _get_obs(self):
        state = np.stack(
            [
                self._agent.movement.x,
                self._agent.movement.y,
                self._targets[self._curr_tgt_id].movement.x,
                self._targets[self._curr_tgt_id].movement.y,
            ]
        )

        return state

    def step(self, shift):
        self._agent.movement.shift(shift[0], shift[1])

        reward = -1 * self._agent.distance_l2(self._targets[self._curr_tgt_id])

        return reward

    def _update_target(self):
        if self._agent.has_collided(self._targets[self._curr_tgt_id]):
            # reward += 5
            if self._curr_tgt_id == len(self._targets) - 1:
                # task solved
                # reward += 100
                self._done = True
            else:
                self._curr_tgt_id += 1


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
        self._agent.movement.shift(shift[0], shift[1])

        reward = -1 * self._agent.distance_l2(self._targets[self._curr_tgt_id])

        self._update_target()

        self._draw_elements_on_canvas()

        obs = self._get_obs()

        self._curr_episode_length += 1
        if self._curr_episode_length == self._max_episode_length:
            self._done = True

        return obs, reward, self._done, {}

    def _update_target(self):
        if self._agent.has_collided(self._targets[self._curr_tgt_id]):
            # reward += 5
            if self._curr_tgt_id == len(self._targets) - 1:
                # task solved
                # reward += 100
                self._done = True
            else:
                self._curr_tgt_id += 1

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

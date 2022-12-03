import numpy as np

from src.ours.env.component.board import EmptyBoard
from src.ours.env.component.point.point import NamedPointWithIcon, PointFactory


class Field:
    def __init__(self, n_targets=2, shift_x=0, shift_y=0, random_init=False):
        self._side_length = 200
        self._board_shape = self._side_length, self._side_length
        self._board = EmptyBoard(self._board_shape)

        self._x_range, self._y_range = self._board.movement_ranges
        self._shift_x, self._shift_y = shift_x, shift_y

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

        self._curr_tgt_id = 0

    def get_pos_agent_target(self):
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

        reward = self._get_reward()

        self._update_target()

        return reward

    def _get_reward(self):
        return -1 * self._agent.distance_l2(self._targets[self._curr_tgt_id])

    def _update_target(self):
        if self._agent.has_collided(self._targets[self._curr_tgt_id]):
            # reward += 5
            if self._curr_tgt_id == len(self._targets) - 1:
                # task solved
                # reward += 100
                self._done = True
            else:
                self._curr_tgt_id += 1

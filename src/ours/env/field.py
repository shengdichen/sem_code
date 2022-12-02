import random

import numpy as np

from src.ours.env.component.point import NamedPointWithIcon, PointFactory


class EmptyBoard:
    def __init__(self, shape: tuple[int, int]):
        self._shape = shape[0], shape[1]

        self._x_range, self._y_range = self._get_movement_ranges()

    def _get_movement_ranges(self) -> tuple[tuple[int, int], tuple[int, int]]:
        y_min = int(self._shape[0] * 0.1)
        x_min = 0
        y_max = int(self._shape[0] * 0.9)
        x_max = self._shape[1]

        return (x_min, x_max), (y_min, y_max)

    @property
    def movement_ranges(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return self._x_range, self._y_range

    def get_reset_agent_pos(self, use_random: bool) -> tuple[int, int]:
        if use_random:
            return self._get_reset_agent_pos_random()
        return self._get_reset_agent_pos_fixed()

    def _get_reset_agent_pos_random(self) -> tuple[int, int]:
        pos_x = random.randrange(int(self._shape[0] * 0.05), int(self._shape[0] * 0.10))
        pos_y = random.randrange(int(self._shape[1] * 0.15), int(self._shape[1] * 0.20))

        return pos_x, pos_y

    @staticmethod
    def _get_reset_agent_pos_fixed() -> tuple[int, int]:
        pos_x = 10
        pos_y = 10

        return pos_x, pos_y

    def get_two_targets_pos_fixed(self, shifts) -> list[tuple[int, int]]:
        shift_x, shift_y = shifts

        pos_target_one = (
            int(self._shape[0] / 2) + shift_x,
            int(self._shape[1] / 2) + shift_y,
        )
        pos_target_two = (
            int(self._shape[0] * 0.95),
            int(self._shape[1] * 0.95),
        )

        pos = [pos_target_one, pos_target_two]
        return pos

    def get_target_pos_random(self) -> tuple[int, int]:
        pos_x = random.randrange(
            self._y_range[0] + int(self._y_range[1] / 4),
            self._y_range[1] - int(self._y_range[1] / 4),
        )
        pos_y = random.randrange(
            self._y_range[0] + int(self._y_range[1] / 4),
            self._y_range[1] - int(self._y_range[1] / 4),
        )

        return pos_x, pos_y


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

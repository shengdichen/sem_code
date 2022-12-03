import numpy as np

from src.ours.env.component.board import EmptyBoard
from src.ours.env.component.point.point import NamedPointWithIcon, PointFactory


class Field:
    def __init__(self, n_targets=2, shifts=(0, 0), random_init=False):
        self._side_length = 200
        self._board = EmptyBoard((self._side_length, self._side_length))

        self._x_range, self._y_range = self._board.movement_ranges
        self._shift_x, self._shift_y = shifts

        self._random_init = random_init

        self._agent = self._make_agent()

        self._n_targets = n_targets
        self._curr_target_id = 0
        self._targets = self._make_targets()

        self._agent_and_targets = []
        self._agent_and_targets.append(self._agent)
        self._agent_and_targets.append(self._targets)

    @property
    def env_config(self):
        return {
            "n_targets": self._n_targets,
            "shift_x": self._shift_x,
            "shift_y": self._shift_y,
        }

    @property
    def agent_and_targets(self):
        return self._agent_and_targets

    @property
    def shape(self):
        return self._side_length, self._side_length

    def _make_agent(self) -> NamedPointWithIcon:
        return PointFactory("agent", self._x_range, self._y_range).create_agent()

    def _make_targets(self, make_random_targets=False) -> list[NamedPointWithIcon]:
        targets = []
        for i in range(self._n_targets):
            target = PointFactory(
                "target_{}".format(i), self._x_range, self._y_range
            ).create_target()

            # TODO: expand to preferences as random process!
            if make_random_targets:
                tgt_x, tgt_y = self._board.get_target_pos_random()
                target.movement.set_position(tgt_x, tgt_y)

            targets.append(target)

        return targets

    def reset(self) -> None:
        pos_x, pos_y = self._board.get_reset_agent_pos(self._random_init)
        self._agent_and_targets[0].movement.set_position(pos_x, pos_y)

        target_positions = self._board.get_two_targets_pos_fixed(
            (self._shift_x, self._shift_y)
        )
        for target, target_pos in zip(self._targets, target_positions):
            target.movement.set_position(target_pos[0], target_pos[1])

        self._curr_target_id = 0

    def get_pos_agent_target(self) -> np.ndarray:
        state = np.stack(
            [
                self._agent_and_targets[0].movement.x,
                self._agent_and_targets[0].movement.y,
                self._targets[self._curr_target_id].movement.x,
                self._targets[self._curr_target_id].movement.y,
            ]
        )

        return state

    def step(self, shift) -> tuple[float, bool]:
        self._agent_and_targets[0].movement.shift(shift[0], shift[1])

        reward = self._get_reward()

        has_visited_all_targets = self._update_target()

        return reward, has_visited_all_targets

    def _get_reward(self) -> float:
        return -1 * self._agent_and_targets[0].distance_l2(
            self._targets[self._curr_target_id]
        )

    def _update_target(self) -> bool:
        has_visited_all_targets = False
        if self._agent_and_targets[0].has_collided(self._targets[self._curr_target_id]):
            # reward += 5
            if self._curr_target_id == len(self._targets) - 1:
                # task solved
                # reward += 100
                has_visited_all_targets = True
            else:
                self._curr_target_id += 1

        return has_visited_all_targets

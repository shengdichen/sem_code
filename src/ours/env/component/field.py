import numpy as np

from src.ours.env.component.board import EmptyBoard
from src.ours.env.component.point.point import NamedPointWithIcon, PointFactory


class Field:
    def __init__(self, n_targets=2, shifts=(0, 0), random_spawn_agent=False):
        self._side_length = 200
        self._board = EmptyBoard((self._side_length, self._side_length))

        self._x_range, self._y_range = self._board.movement_ranges
        self._shifts_first_default_target = shifts

        self._random_spawn_agent = random_spawn_agent
        self._use_random_targets = False

        self._n_targets = n_targets
        self._curr_target_id = 0

        self._agent_and_targets = [self._make_agent(), self._make_targets()]

    @property
    def env_config(self):
        return {
            "n_targets": self._n_targets,
            "shift_x": self._shifts_first_default_target[0],
            "shift_y": self._shifts_first_default_target[1],
        }

    @property
    def agent_and_targets(self):
        return self._agent_and_targets

    @property
    def shape(self):
        return self._side_length, self._side_length

    def _make_agent(self) -> NamedPointWithIcon:
        agent = PointFactory("agent", self._x_range, self._y_range).create_agent()
        agent.movement.set_position(
            self._board.get_reset_agent_pos(self._random_spawn_agent)
        )
        return agent

    def _make_targets(self) -> list[NamedPointWithIcon]:
        targets = [
            PointFactory(
                "target_{}".format(i), self._x_range, self._y_range
            ).create_target()
            for i in range(self._n_targets)
        ]

        self._set_targets_pos(targets)

        return targets

    def _set_targets_pos(self, targets: list[NamedPointWithIcon]) -> None:
        # TODO: expand to preferences as random process!
        if self._use_random_targets:
            self._set_targets_position_random(targets)
        elif self._n_targets == 2:
            self._set_targets_position_fixed(targets)
        else:
            exit(1)

    def _set_targets_position_random(self, targets: list[NamedPointWithIcon]) -> None:
        for target in targets:
            target.movement.set_position(self._board.get_target_pos_random())

    def _set_targets_position_fixed(self, targets: list[NamedPointWithIcon]) -> None:
        target_positions = self._board.get_two_targets_pos_fixed(
            self._shifts_first_default_target
        )
        for target, target_pos in zip(targets, target_positions):
            target.movement.set_position(target_pos)

    def reset(self) -> None:
        self._agent_and_targets[0].movement.set_position(
            self._board.get_reset_agent_pos(self._random_spawn_agent)
        )

        if self._use_random_targets:
            self._set_targets_position_random(self.agent_and_targets[1])
        else:
            pass  # pos of fixed targets are already set in c'tor

        self._curr_target_id = 0

    def get_pos_agent_target(self) -> np.ndarray:
        state = np.stack(
            [
                self._agent_and_targets[0].movement.x,
                self._agent_and_targets[0].movement.y,
                self._agent_and_targets[1][self._curr_target_id].movement.x,
                self._agent_and_targets[1][self._curr_target_id].movement.y,
            ]
        )

        return state

    def step(self, action: tuple[int, int]) -> tuple[float, bool]:
        self._agent_and_targets[0].movement.shift(action[0], action[1])

        reward = self._get_reward()

        has_visited_all_targets = self._update_target()

        return reward, has_visited_all_targets

    def _get_reward(self) -> float:
        return -1 * self._agent_and_targets[0].distance_l2(
            self._agent_and_targets[1][self._curr_target_id]
        )

    def _update_target(self) -> bool:
        has_visited_all_targets = False
        if self._agent_and_targets[0].has_collided(
            self._agent_and_targets[1][self._curr_target_id]
        ):
            # reward += 5
            if self._curr_target_id == len(self._agent_and_targets[1]) - 1:
                # task solved
                # reward += 100
                has_visited_all_targets = True
            else:
                self._curr_target_id += 1

        return has_visited_all_targets

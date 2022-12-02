import random


class MovementField:
    def __init__(self, shape: tuple[int, int]):
        self._shape = shape[0], shape[1]

        (
            self._y_min,
            self._x_min,
            self._y_max,
            self._x_max,
        ) = self._get_movement_ranges()

    def _get_movement_ranges(self) -> tuple[int, int, int, int]:
        y_min = int(self._shape[0] * 0.1)
        x_min = 0
        y_max = int(self._shape[0] * 0.9)
        x_max = self._shape[1]

        return y_min, x_min, y_max, x_max

    @property
    def movement_ranges(self) -> tuple[int, int, int, int]:
        return self._y_min, self._x_min, self._y_max, self._x_max

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
            self._y_min + int(self._y_max / 4), self._y_max - int(self._y_max / 4)
        )
        pos_y = random.randrange(
            self._y_min + int(self._y_max / 4), self._y_max - int(self._y_max / 4)
        )

        return pos_x, pos_y

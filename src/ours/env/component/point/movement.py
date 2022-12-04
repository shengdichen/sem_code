import numpy as np


class MovementTwoDim:
    def __init__(self, x_min_max_with_icon, y_min_max_with_icon):
        self._x_movement = MovementOneDim(x_min_max_with_icon)
        self._y_movement = MovementOneDim(y_min_max_with_icon)

    def set_position(self, pos_desired: tuple[float, float]) -> None:
        pos_desired_x, pos_desired_y = pos_desired
        self._x_movement.set(pos_desired_x)
        self._y_movement.set(pos_desired_y)

    def get_position(self) -> tuple[float, float]:
        return self._x_movement.pos, self._y_movement.pos

    def shift(self, shift_by_x: float, shift_by_y: float) -> None:
        self._x_movement.shift(shift_by_x)
        self._y_movement.shift(shift_by_y)

    def distance_l2(self, that: "MovementTwoDim") -> float:
        return np.sqrt(
            self._x_movement.diff_squared(that._x_movement)
            + self._y_movement.diff_squared(that._y_movement)
        )


class MovementOneDim:
    def __init__(self, pos_range: tuple[float, float]):
        self._pos = 0.0
        self._pos_min, self._pos_max = pos_range

    @property
    def pos(self):
        return self._pos

    def set(self, pos_desired: float) -> None:
        self._pos = self._clamp(pos_desired)

    def shift(self, shift_by: float) -> None:
        self.set(self._pos + shift_by)

    def get_range(self) -> tuple[float, float]:
        return self._pos_min, self._pos_max

    def _clamp(self, pos_desired: float) -> float:
        return max(min(self._pos_max, pos_desired), self._pos_min)

    def diff_squared(self, that: "MovementOneDim") -> float:
        return (self._pos - that._pos) ** 2

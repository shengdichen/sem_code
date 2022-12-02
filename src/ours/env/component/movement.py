import numpy as np


class MovementTwoDim:
    def __init__(self, x_min_max_with_icon, y_min_max_with_icon):
        self.x_min, self.x_max_with_icon = x_min_max_with_icon
        self.y_min, self.y_max_with_icon = y_min_max_with_icon

        self.x_movement = MovementOneDim((self.x_min, self.x_max_with_icon))
        self.y_movement = MovementOneDim((self.y_min, self.y_max_with_icon))

    @property
    def x(self):
        return self.x_movement.pos

    @property
    def y(self):
        return self.y_movement.pos

    def set_position(self, pos_desired_x: float, pos_desired_y: float) -> None:
        self.x_movement.set(pos_desired_x)
        self.y_movement.set(pos_desired_y)

    def get_position(self) -> tuple[float, float]:
        return self.x, self.y

    def shift(self, shift_by_x: float, shift_by_y: float) -> None:
        self.x_movement.shift(shift_by_x)
        self.y_movement.shift(shift_by_y)

    def distance_l2(self, that: "MovementTwoDim") -> float:
        return np.sqrt(
            self.x_movement.diff_squared(that.x_movement)
            + self.y_movement.diff_squared(that.y_movement)
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

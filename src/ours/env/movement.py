class Point:
    def __init__(self, x_max_with_icon, x_min, y_max_with_icon, y_min):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max_with_icon = x_max_with_icon
        self.y_max_with_icon = y_max_with_icon

        self.x_movement = MovementOneDim((self.x_min, self.x_max_with_icon))
        self.y_movement = MovementOneDim((self.y_min, self.y_max_with_icon))

    @property
    def x(self):
        return self.x_movement.pos

    @property
    def y(self):
        return self.y_movement.pos

    def set_position(self, x: float, y: float) -> None:
        self.x_movement.set(x)
        self.y_movement.set(y)

    def get_position(self) -> tuple[float, float]:
        return self.x, self.y

    def move(self, del_x: float, del_y: float) -> None:
        self.x_movement.shift(del_x)
        self.y_movement.shift(del_y)


class MovementOneDim:
    def __init__(self, pos_range: tuple[float, float]):
        self._pos = 0
        self._pos_min, self._pos_max = pos_range

    @property
    def pos(self):
        return self._pos

    def set(self, value: float) -> None:
        self._pos = self._clamp(value)

    def shift(self, shift_by: float) -> None:
        self.set(self._pos + shift_by)

    def get_range(self) -> tuple[float, float]:
        return self._pos_min, self._pos_max

    def _clamp(self, pos_desired: float) -> float:
        return max(min(self._pos_max, pos_desired), self._pos_min)

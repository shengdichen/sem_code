from src.ours.env.icon import Icon


class Point:
    def __init__(self, x_max_with_icon, x_min, y_max_with_icon, y_min):
        self.x = 0
        self.y = 0

        self.x_min = x_min
        self.y_min = y_min
        self.x_max_with_icon = x_max_with_icon
        self.y_max_with_icon = y_max_with_icon

    def set_position(self, x, y):
        self.x = self.clamp(x, self.x_min, self.x_max_with_icon)
        self.y = self.clamp(y, self.y_min, self.y_max_with_icon)

    def get_position(self):
        return self.x, self.y

    def move(self, del_x, del_y):
        self.x += del_x
        self.y += del_y

        self.x = self.clamp(self.x, self.x_min, self.x_max_with_icon)
        self.y = self.clamp(self.y, self.y_min, self.y_max_with_icon)

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)


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

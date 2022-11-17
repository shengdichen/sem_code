from src.ours.env.component.icon.icon import Icon
from src.ours.env.component.movement import MovementTwoDim


class NamedPointWithIcon:
    def __init__(
        self, name, range_x_without_icon, range_y_without_icon, icon_path, icon_size
    ):
        self._name = name

        self.x_min, self.x_max = range_x_without_icon
        self.y_min, self.y_max = range_y_without_icon

        self.x_icon, self.y_icon = icon_size

        self._x_max_with_icon, self._y_max_with_icon = (
            self.x_max - self.x_icon,
            self.y_max - self.y_icon,
        )
        self.movement = MovementTwoDim(
            self._x_max_with_icon, self.x_min, self._y_max_with_icon, self.y_min
        )
        self.icon = Icon(icon_path, icon_size).icon

    def has_collided(self, that: "NamedPointWithIcon"):
        has_collided_x, has_collided_y = False, False

        this_x, this_y = self.movement.get_position()
        that_x, that_y = that.movement.get_position()

        if 2 * abs(this_x - that_x) <= (self.x_icon + that.x_icon):
            has_collided_x = True

        if 2 * abs(this_y - that_y) <= (self.y_icon + that.y_icon):
            has_collided_y = True

        return has_collided_x and has_collided_y


class PointFactory:
    def __init__(self, name: str, x_max, x_min, y_max, y_min):
        self.name = name
        self.x_max, self.x_min = x_max, x_min
        self.y_max, self.y_min = y_max, y_min

    def create_agent(self) -> NamedPointWithIcon:
        return NamedPointWithIcon(
            self.name,
            (self.x_min, self.x_max),
            (self.y_min, self.y_max),
            "./agent.png",
            (5, 5),
        )

    def create_target(self) -> NamedPointWithIcon:
        return NamedPointWithIcon(
            self.name,
            (self.x_min, self.x_max),
            (self.y_min, self.y_max),
            "./star.png",
            (4, 4),
        )

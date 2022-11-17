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

    def has_collided(elem1: "NamedPointWithIcon", elem2: "NamedPointWithIcon"):
        x_col = False
        y_col = False

        elem1_x, elem1_y = elem1.movement.get_position()
        elem2_x, elem2_y = elem2.movement.get_position()

        if 2 * abs(elem1_x - elem2_x) <= (elem1.x_icon + elem2.x_icon):
            x_col = True

        if 2 * abs(elem1_y - elem2_y) <= (elem1.y_icon + elem2.y_icon):
            y_col = True

        if x_col and y_col:
            return True

        return False


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

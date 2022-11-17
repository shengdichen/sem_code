from src.ours.env.icon import Icon
from src.ours.env.movement import MovementTwoDim


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
        self.point = MovementTwoDim(
            self._x_max_with_icon, self.x_min, self._y_max_with_icon, self.y_min
        )
        self.icon = Icon(icon_path, icon_size).icon


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
            "agent.png",
            (5, 5),
        )

    def create_target(self) -> NamedPointWithIcon:
        return NamedPointWithIcon(
            self.name,
            (self.x_min, self.x_max),
            (self.y_min, self.y_max),
            "star.png",
            (4, 4),
        )


class Agent(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        self.name = name

        self.icon_w, self.icon_h = 5, 5

        super(Agent, self).__init__(
            x_max, x_min, y_max, y_min, self.icon_w, self.icon_h
        )

        self.icon = Icon("agent.png", (self.icon_w, self.icon_h)).icon


class Target(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        self.name = name

        self.icon_w, self.icon_h = 4, 4

        super(Target, self).__init__(
            x_max, x_min, y_max, y_min, self.icon_w, self.icon_h
        )

        self.icon = Icon("star.png", (self.icon_w, self.icon_h)).icon

from src.ours.env.icon import Icon
from src.ours.env.movement import Point


class Agent(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        self.icon_w, self.icon_h = 5, 5

        super(Agent, self).__init__(
            name, x_max, x_min, y_max, y_min, self.icon_w, self.icon_h
        )

        self.icon = Icon("agent.png", (self.icon_w, self.icon_h)).icon


class Target(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        self.icon_w, self.icon_h = 4, 4

        super(Target, self).__init__(
            name, x_max, x_min, y_max, y_min, self.icon_w, self.icon_h
        )

        self.icon = Icon("star.png", (self.icon_w, self.icon_h)).icon

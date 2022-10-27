import cv2


class Point(object):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        self.x = 0
        self.y = 0
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.name = name

    def set_position(self, x, y):
        self.x = self.clamp(x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(y, self.y_min, self.y_max - self.icon_h)

    def get_position(self):
        return (self.x, self.y)

    def move(self, del_x, del_y):
        self.x += del_x
        self.y += del_y

        self.x = self.clamp(self.x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(self.y, self.y_min, self.y_max - self.icon_h)

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)


class Agent(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Agent, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("agent.png") / 255.0
        # self.icon = cv2.circle(image, center_coordinates, radius, color, thickness)

        self.icon_w = 5
        self.icon_h = 5
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))


class Target(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Target, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("star.png") / 255.0
        self.icon_w = 4
        self.icon_h = 4
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))

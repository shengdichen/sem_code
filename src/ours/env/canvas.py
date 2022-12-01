from abc import abstractmethod, ABC

import numpy as np

from src.ours.env.component.point import NamedPointWithIcon


class VisualizerBase(ABC):
    def __init__(self, shape: tuple[int, int]):
        self._colormat_shape = shape[0], shape[1], 3

    @abstractmethod
    def register(self, **kwargs):
        pass

    def get_ranges(self):
        y_min = int(self._colormat_shape[0] * 0.1)
        x_min = 0
        y_max = int(self._colormat_shape[0] * 0.9)
        x_max = self._colormat_shape[1]

        return y_min, x_min, y_max, x_max


class AgentTargetsVisualizer(VisualizerBase):
    def __init__(self, shape: tuple[int, int]):
        super().__init__(shape)

        self._colormat = self._make()

    @property
    def colormat(self):
        return self._colormat

    def _make(self) -> np.ndarray:
        return np.ones(self._colormat_shape)

    def register(self, points: list[NamedPointWithIcon]) -> None:
        self._colormat = self._make()

        for point in points:
            self._colormat[
                point.movement.y : point.movement.y + point.y_icon,
                point.movement.x : point.movement.x + point.x_icon,
            ] = point.icon

        # text = 'Time Left: {} | Rewards: {}'.format(self.time, self.ep_return)

        # Put the info on canvas
        # self.canvas = cv2.putText(
        #     self.canvas, text, (10, 20), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA
        # )


class TrajectoryHeatVisualizer(VisualizerBase):
    def __init__(self, shape: tuple[int, int]):
        super().__init__(shape)

        self._colormat = self._make()

    @property
    def colormat(self):
        return self._colormat

    def _make(self) -> np.ndarray:
        return np.zeros(self._colormat_shape)

    def register(self, point: NamedPointWithIcon) -> None:
        self._colormat[
            point.movement.y : point.movement.y + point.y_icon,
            point.movement.x : point.movement.x + point.x_icon,
        ] += 1

        # normalize hist canvas
        # self.canvas_hist = self.canvas_hist / np.sum(self.canvas_hist)

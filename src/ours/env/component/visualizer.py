from abc import abstractmethod, ABC

import numpy as np

from src.ours.env.component.field import Field
from src.ours.env.component.point.point import NamedPointWithIcon


class VisualizerBase(ABC):
    def __init__(self, field: Field):
        self._field = field
        self._colormat_shape = field.shape[0], field.shape[1], 3

    @abstractmethod
    def register_all(self, **kwargs):
        pass


class PositionVisualizer(VisualizerBase):
    def __init__(self, field: Field):
        super().__init__(field)

        self._colormat = self._make()

    @property
    def colormat(self):
        return self._colormat

    def _make(self) -> np.ndarray:
        return np.ones(self._colormat_shape)

    def register_all(self):
        self._register(self._field._agent_and_targets)

    def _register(self, points: list[NamedPointWithIcon]) -> None:
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
    def __init__(self, field: Field):
        super().__init__(field)

        self._colormat = self._make()

    @property
    def colormat(self):
        return self._colormat

    def _make(self) -> np.ndarray:
        return np.zeros(self._colormat_shape)

    def register_all(self):
        self._register(self._field._agent)

    def _register(self, point: NamedPointWithIcon) -> None:
        self._colormat[
            point.movement.y : point.movement.y + point.y_icon,
            point.movement.x : point.movement.x + point.x_icon,
        ] += 1

        # normalize hist canvas
        # self.canvas_hist = self.canvas_hist / np.sum(self.canvas_hist)

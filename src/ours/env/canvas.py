from abc import abstractmethod, ABC

import numpy as np

from src.ours.env.component.point import NamedPointWithIcon


class VisuaizerBase(ABC):
    def __init__(self, shape: tuple[int, int]):
        self._colormat_shape = shape[0], shape[1], 3

    @abstractmethod
    def register(self, **kwargs):
        pass


class AgentTargetsVisualizer(VisuaizerBase):
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


class TrajectoryHeatVisualizer(VisuaizerBase):
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


class Visualizer:
    def __init__(self, canvas_shape: tuple[int, int]):
        self._colormap_shape = canvas_shape[0], canvas_shape[1], 3

        self._canvas, self._canvas_hist = self._make_canvas_and_hist()

    @property
    def canvas(self):
        return self._canvas

    @property
    def canvas_hist(self):
        return self._canvas_hist

    def _make_canvas_and_hist(self) -> tuple[np.ndarray, np.ndarray]:
        return np.ones(self._colormap_shape), np.zeros(self._colormap_shape)

    def register_on_canvas(self, points: list[NamedPointWithIcon]) -> None:
        self._canvas = np.ones(self._colormap_shape)

        for point in points:
            self._canvas[
                point.movement.y : point.movement.y + point.y_icon,
                point.movement.x : point.movement.x + point.x_icon,
            ] = point.icon

        # text = 'Time Left: {} | Rewards: {}'.format(self.time, self.ep_return)

        # Put the info on canvas
        # self.canvas = cv2.putText(
        #     self.canvas, text, (10, 20), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA
        # )

    def register_on_hist(self, point: NamedPointWithIcon) -> None:
        self._canvas_hist[
            point.movement.y : point.movement.y + point.y_icon,
            point.movement.x : point.movement.x + point.x_icon,
        ] += 1

        # normalize hist canvas
        # self.canvas_hist = self.canvas_hist / np.sum(self.canvas_hist)

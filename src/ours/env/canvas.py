import random
from abc import abstractmethod, ABC

import numpy as np

from src.ours.env.component.point import NamedPointWithIcon


class MovementField:
    def __init__(self, shape: tuple[int, int]):
        self._shape = shape[0], shape[1]

        (
            self._y_min,
            self._x_min,
            self._y_max,
            self._x_max,
        ) = self.get_movement_ranges()

    def get_movement_ranges(self):
        y_min = int(self._shape[0] * 0.1)
        x_min = 0
        y_max = int(self._shape[0] * 0.9)
        x_max = self._shape[1]

        return y_min, x_min, y_max, x_max

    def get_reset_agent_pos(self, use_random):
        if use_random:
            return self._get_reset_agent_pos_random()
        return self._get_reset_agent_pos_fixed()

    def _get_reset_agent_pos_random(self):
        x = random.randrange(
            int(self._shape[0] * 0.05), int(self._shape[0] * 0.10)
        )
        y = random.randrange(
            int(self._shape[1] * 0.15), int(self._shape[1] * 0.20)
        )

        return x, y

    def _get_reset_agent_pos_fixed(self):
        x = 10
        y = 10

        return x, y

    def get_reset_targets_pos(self, shifts):
        shift_x, shift_y = shifts

        pos_target_one = (
            int(self._shape[0] / 2) + shift_x,
            int(self._shape[1] / 2) + shift_y,
        )
        pos_target_two = (
            int(self._shape[0] * 0.95),
            int(self._shape[1] * 0.95),
        )

        pos = [pos_target_one, pos_target_two]
        return pos

    def get_target_pos_random(self):
        tgt_x = random.randrange(
            self._y_min + int(self._y_max / 4), self._y_max - int(self._y_max / 4)
        )
        tgt_y = random.randrange(
            self._y_min + int(self._y_max / 4), self._y_max - int(self._y_max / 4)
        )

        return tgt_x, tgt_y


class VisualizerBase(ABC):
    def __init__(self, shape: tuple[int, int]):
        self._colormat_shape = shape[0], shape[1], 3

    @abstractmethod
    def register(self, **kwargs):
        pass


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

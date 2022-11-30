from abc import ABC, abstractmethod

import cv2
import numpy as np
from matplotlib import pyplot as plt


class PointEnvRendererBase(ABC):
    @abstractmethod
    def render(self) -> None:
        pass


class PointEnvRendererHuman(PointEnvRendererBase):
    def __init__(self, canvas: np.ndarray, canvas_hist: np.ndarray):
        self._canvas = canvas

        self._heatmap = self._get_heatmap(canvas_hist)
        self._separator = self._get_separator()

    @staticmethod
    def _get_heatmap(canvas_hist: np.ndarray):
        heatmapimg = np.array(canvas_hist * 255, dtype=np.uint8)
        heatmap = cv2.applyColorMap(heatmapimg, cv2.COLORMAP_JET)
        heatmap = heatmap / 255

        return heatmap

    @staticmethod
    def _get_separator() -> np.ndarray:
        return np.ones([200, 4, 3]) * 0.2

    def render(self) -> None:
        cat_img = self._get_image()
        cv2.imshow("game", cat_img)
        # plt.imshow("Game", cat_img)
        cv2.waitKey(10)

    def _get_image(self) -> np.ndarray:
        return np.hstack((self._canvas, self._separator, self._heatmap))

    @staticmethod
    def clean_up() -> None:
        cv2.destroyAllWindows()
        plt.close("all")


class PointEnvRendererRgb(PointEnvRendererBase):
    def __init__(self, canvas: np.ndarray):
        self._canvas = canvas

    def render(self) -> None:
        print(self._canvas)

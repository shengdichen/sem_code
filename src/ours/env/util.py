from abc import ABC, abstractmethod

import cv2
import numpy as np
from matplotlib import pyplot as plt


class PointEnvRendererBase(ABC):
    @abstractmethod
    def render(self) -> None:
        pass


class PointEnvRenderer(PointEnvRendererBase):
    def __init__(self, canvas: np.ndarray, canvas_hist: np.ndarray):
        self._canvas = canvas
        self._canvas_hist = canvas_hist

    def render(self) -> None:
        cat_img = self._get_image()
        cv2.imshow("game", cat_img)
        # plt.imshow("Game", cat_img)
        cv2.waitKey(10)

    def _get_image(self):
        heatmap = self._get_heatmap()
        separator = self._get_separator()
        return np.hstack((self._canvas, separator, heatmap))

    def _get_heatmap(self):
        heatmapimg = np.array(self._canvas_hist * 255, dtype=np.uint8)
        heatmap = cv2.applyColorMap(heatmapimg, cv2.COLORMAP_JET)
        heatmap = heatmap / 255

        return heatmap

    @staticmethod
    def _get_separator():
        return np.ones([200, 4, 3]) * 0.2

    @staticmethod
    def clean_up():
        cv2.destroyAllWindows()
        plt.close("all")


class PointEnvRendererRgb(PointEnvRendererBase):
    def __init__(self, canvas: np.ndarray):
        self._canvas = canvas

    def render(self) -> None:
        print(self._canvas)

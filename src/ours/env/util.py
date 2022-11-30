import cv2
import numpy as np


class PointEnvRenderer:
    def __init__(self, canvas: np.ndarray, canvas_hist: np.ndarray):
        self._canvas = canvas
        self._canvas_hist = canvas_hist

    def render(self, mode: str) -> None:
        if mode == "human":
            self.render_human()
        else:
            self.render_rgb()

    def render_human(self) -> None:
        heatmap = self._get_heatmap()
        separator = self._get_separator()
        cat_img = np.hstack((self._canvas, separator, heatmap))
        cv2.imshow("game", cat_img)
        # plt.imshow("Game", cat_img)
        cv2.waitKey(10)

    def _get_heatmap(self):
        heatmapimg = np.array(self._canvas_hist * 255, dtype=np.uint8)
        heatmap = cv2.applyColorMap(heatmapimg, cv2.COLORMAP_JET)
        heatmap = heatmap / 255

        return heatmap

    @staticmethod
    def _get_separator():
        return np.ones([200, 4, 3]) * 0.2

    def render_rgb(self) -> None:
        print(self._canvas)

    @staticmethod
    def clean_up():
        cv2.destroyAllWindows()

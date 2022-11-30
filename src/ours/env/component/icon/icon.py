from pathlib import Path

import cv2


class Icon:
    def __init__(self, path_rel: str, size: tuple[int, int]):
        self._icon = self._load(path_rel)

        self._size = size
        self._resize()

    @property
    def icon(self):
        return self._icon

    @property
    def size(self):
        return self._size

    @staticmethod
    def _load(path_rel: str):
        curr_dir = Path(__file__).absolute().parent
        path_abs = curr_dir / path_rel
        return cv2.imread(str(path_abs)) / 255.0

    def _make_square_icon(self):
        # self._icon = cv2.circle(image, center_coordinates, radius, color, thickness)
        pass

    def _resize(self):
        self._icon = cv2.resize(self._icon, self._size)

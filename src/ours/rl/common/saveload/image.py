from pathlib import Path

import numpy as np
from PIL import Image


class ImageSaveLoad:
    def __init__(self, path: Path):
        self._path = str(path)
        self._make_path_image_usable()

    def _make_path_image_usable(self) -> None:
        actual_suffix, image_suffix = self._path[-4:], ".png"
        if actual_suffix != image_suffix:
            self._path += image_suffix

    def save_from_np(self, target: np.ndarray, force_resave: bool = False) -> None:
        im = Image.fromarray(target)
        self.save(im, force_resave)

    def save(self, target: Image.Image, force_resave: bool = False) -> None:
        if force_resave:
            target.save(self._path)
        elif not Path.exists(Path(self._path)):
            target.save(self._path)
        else:
            return

    def load(self) -> Image.Image:
        return Image.open(self._path)

    def exists(self) -> bool:
        try:
            self.load()
        except FileNotFoundError:
            return False
        else:
            return True

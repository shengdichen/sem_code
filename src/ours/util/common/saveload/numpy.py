from pathlib import Path

import numpy as np


class NumpySaveLoad:
    def __init__(self, path: Path):
        self._path = str(path)
        self._make_path_numpy_usable()

    def _make_path_numpy_usable(self) -> None:
        actual_suffix, numpy_suffix = self._path[-4:], ".npy"
        if actual_suffix != numpy_suffix:
            self._path += numpy_suffix

    def save(self, target: np.ndarray, force_resave: bool = False) -> None:
        if force_resave:
            np.save(self._path, target)
        elif not Path.exists(Path(self._path)):
            np.save(self._path, target)
        else:
            return

    def load(self) -> np.ndarray:
        return np.load(self._path)

    def exists(self) -> bool:
        try:
            self.load()
        except FileNotFoundError:
            return False
        else:
            return True

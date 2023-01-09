from pathlib import Path

import numpy as np


class NumpySaveLoad:
    def __init__(self, path: Path):
        self._path = str(path)

    def save(self, target: np.ndarray) -> None:
        np.save(self._path, target)

    def load(self) -> np.ndarray:
        return np.load(self._path)

    def exists(self) -> bool:
        try:
            self.load()
        except FileNotFoundError:
            return False
        else:
            return True

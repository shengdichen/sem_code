from pathlib import Path

import numpy as np

from src.ours.rl.common.saveload.numpy import NumpySaveLoad


class TrajectorySaveLoad:
    def __init__(self, path: Path):
        self._saveloader = NumpySaveLoad(path)

    def save(self, target: np.ndarray) -> None:
        self._saveloader.save(target)

    def load(self) -> np.ndarray:
        return self._saveloader.load()

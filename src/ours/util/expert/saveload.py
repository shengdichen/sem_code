from pathlib import Path

import numpy as np


class ExpertSaveLoad:
    def __init__(self, path: Path):
        self._path = str(path)

    def save(self, target):
        np.save(self._path, target)

    def load(self):
        return np.load(self._path)

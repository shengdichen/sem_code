from pathlib import Path

import numpy as np

from src.ours.util.common.saveload import NumpySaveLoad


class TestNumpySaveLoad:
    @staticmethod
    def setup() -> tuple[np.ndarray, Path, Path]:
        target = np.array([1, 2])
        path_raw = Path("test_save")
        path_numpy = Path(str(path_raw) + ".npy")

        return target, path_raw, path_numpy

    @staticmethod
    def remove_file(filename: Path) -> None:
        Path.unlink(filename)

    def test_save(self):
        target, path_raw, path_numpy = self.setup()

        NumpySaveLoad(path_raw).save(target)
        assert Path.exists(path_numpy)
        self.remove_file(path_numpy)

        NumpySaveLoad(path_numpy).save(target)
        assert Path.exists(path_numpy)
        self.remove_file(path_numpy)

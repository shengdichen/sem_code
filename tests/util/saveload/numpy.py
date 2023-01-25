from pathlib import Path

import numpy as np

from src.ours.util.saveload.numpy import NumpySaveLoad


class TestNumpySaveLoad:
    @staticmethod
    def setup() -> tuple[np.ndarray, Path, Path]:
        target = np.array([1, 2])
        path_raw = Path("test_save")
        path_numpy = Path(str(path_raw) + ".npy")

        return target, path_raw, path_numpy

    @staticmethod
    def cleanup(filename: Path) -> None:
        Path.unlink(filename)

    def test_save(self) -> None:
        target, path_raw, path_numpy = self.setup()

        NumpySaveLoad(path_raw).save(target)
        assert Path.exists(path_numpy)
        self.cleanup(path_numpy)

        NumpySaveLoad(path_numpy).save(target)
        assert Path.exists(path_numpy)
        self.cleanup(path_numpy)

    def test_should_exist(self) -> None:
        target, path_raw, path_numpy = self.setup()

        saveloader = NumpySaveLoad(path_raw)
        saveloader.save(target)
        assert saveloader.exists()
        self.cleanup(path_numpy)

        saveloader = NumpySaveLoad(path_numpy)
        saveloader.save(target)
        assert saveloader.exists()
        self.cleanup(path_numpy)

    def test_should_not_exist(self) -> None:
        target, path_raw, path_numpy = self.setup()

        NumpySaveLoad(path_raw).save(target)

        wrong_path_raw, wrong_path_numpy = Path("wrong_file_name"), Path(
            "wrong_file_name.npy"
        )
        assert not NumpySaveLoad(wrong_path_raw).exists()
        assert not NumpySaveLoad(wrong_path_numpy).exists()

        self.cleanup(path_numpy)

    def test_load(self) -> None:
        target, path_raw, path_numpy = self.setup()

        saveloader = NumpySaveLoad(path_raw)
        saveloader.save(target)
        assert np.all(saveloader.load() == target)
        self.cleanup(path_numpy)

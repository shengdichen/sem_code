from abc import ABC, abstractmethod

from src.ours.env.env import MovePoint


class EnvFactory(ABC):
    @abstractmethod
    def create(self):
        pass


class PointEnvFactory(EnvFactory):
    def __init__(self, env_config: dict[str:int]):
        self._env_config = env_config

    def create(self):
        return MovePoint(**self._env_config)


class PathGenerator:
    def __init__(self, env_config: dict[str:int]):
        self._env_config = env_config

        self._prefix = "exp"
        self._connector = "_"

    def get_filename_from_shift_values(self) -> str:
        shift_x, shift_y = self._env_config["shift_x"], self._env_config["shift_y"]
        return (
            self._prefix
            + self._connector
            + str(shift_x)
            + self._connector
            + str(shift_y)
        )

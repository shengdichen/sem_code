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


class PointEnvIdentifierGenerator:
    def __init__(self, env_config: dict[str:int]):
        self._env_config = env_config

        self._prefix = "pointenv"
        self._connector = "_"

    def get_identifier(self) -> str:
        shift_x, shift_y = self._env_config["shift_x"], self._env_config["shift_y"]
        return (
            self._prefix
            + self._connector
            + "{0:03}".format(shift_x)
            + self._connector
            + "{0:03}".format(shift_y)
        )

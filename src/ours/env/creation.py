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


class PointEnvConfigFactory:
    def __init__(self):
        self._env_configs = [
            {"n_targets": 2, "shift_x": 0, "shift_y": 0},
            {"n_targets": 2, "shift_x": 0, "shift_y": 50},
            {"n_targets": 2, "shift_x": 50, "shift_y": 0},
        ]

    @property
    def env_configs(self):
        return self._env_configs


class PointEnvIdentifierGenerator:
    def __init__(self):
        self._prefix = "pointenv"
        self._connector = "_"

    def get_identifier(self, env_config: dict[str:int]) -> str:
        shift_x, shift_y = env_config["shift_x"], env_config["shift_y"]
        return (
            self._prefix
            + self._connector
            + "{0:03}".format(shift_x)
            + self._connector
            + "{0:03}".format(shift_y)
        )

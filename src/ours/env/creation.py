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

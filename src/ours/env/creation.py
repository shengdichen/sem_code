from abc import ABC, abstractmethod

from gym import Env

from src.ours.env.env import MovePoint, DiscreteMovePoint, ContMovePoint


class EnvFactory(ABC):
    @abstractmethod
    def create(self) -> Env:
        pass


class PointEnvFactory(EnvFactory):
    def __init__(self, env_config: dict[str:int]):
        self._env_config = env_config

    def create(self) -> MovePoint:
        pass


class DiscretePointEnvFactory(PointEnvFactory):
    def __init__(self, env_config: dict[str:int]):
        super().__init__(env_config)

    def create(self) -> DiscreteMovePoint:
        return DiscreteMovePoint(**self._env_config)


class ContPointEnvFactory(PointEnvFactory):
    def __init__(self, env_config: dict[str:int]):
        super().__init__(env_config)

    def create(self) -> ContMovePoint:
        return ContMovePoint(**self._env_config)


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

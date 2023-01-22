from abc import ABC, abstractmethod

from gym import Env

from src.ours.env.env import MovePointBase, MovePoint, MovePointCont


class EnvFactory(ABC):
    @abstractmethod
    def create(self) -> Env:
        pass


class PointEnvFactoryBase(EnvFactory):
    def __init__(self, env_config: dict[str:int]):
        self._env_config = env_config

    def create(self) -> MovePointBase:
        pass


class PointEnvFactory(PointEnvFactoryBase):
    def __init__(self, env_config: dict[str:int]):
        super().__init__(env_config)

    def create(self) -> MovePoint:
        return MovePoint(**self._env_config)


class PointEnvContFactory(PointEnvFactoryBase):
    def __init__(self, env_config: dict[str:int]):
        super().__init__(env_config)

    def create(self) -> MovePointCont:
        return MovePointCont(**self._env_config)


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


class PointEnvIdentifierGeneratorBase:
    def __init__(self, prefix: str):
        self._prefix = prefix
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

    def from_env(self, env: MovePoint) -> str:
        return self.get_identifier(env.env_config)


class PointEnvIdentifierGenerator(PointEnvIdentifierGeneratorBase):
    def __init__(self):
        super().__init__("pointenv")


class PointEnvContIdentifierGenerator(PointEnvIdentifierGeneratorBase):
    def __init__(self):
        super().__init__("pointenv_cont")

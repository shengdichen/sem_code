from abc import ABC, abstractmethod

from gym import Env

from src.ours.env.env import PointNav, DiscretePointNav, ContPointNav


class EnvFactory(ABC):
    @abstractmethod
    def create(self) -> Env:
        pass


class PointNavFactory(EnvFactory):
    def __init__(self, env_config: dict[str:int]):
        self._env_config = env_config

    def create(self) -> PointNav:
        pass


class DiscretePointNavFactory(PointNavFactory):
    def __init__(self, env_config: dict[str:int]):
        super().__init__(env_config)

    def create(self) -> DiscretePointNav:
        return DiscretePointNav(**self._env_config)


class ContPointNavFactory(PointNavFactory):
    def __init__(self, env_config: dict[str:int]):
        super().__init__(env_config)

    def create(self) -> ContPointNav:
        return ContPointNav(**self._env_config)

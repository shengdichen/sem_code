from abc import ABC, abstractmethod


class EnvFactory(ABC):
    @abstractmethod
    def create(self):
        pass

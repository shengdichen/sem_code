import random

import numpy as np


class ActionProvider:
    def get_action(self, obs: np.ndarray, **kwargs) -> int | np.ndarray:
        pass


class ActionProviderRandom(ActionProvider):
    def get_action(self, obs: np.ndarray, **kwargs) -> int:
        return random.randint(0, 4)


class ActionProviderRandomCont(ActionProvider):
    def get_action(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        return np.array(np.random.uniform([-2.5, -2.5], [+2.5, +2.5]))

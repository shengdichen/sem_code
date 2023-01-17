import random

import numpy as np


class ActionProvider:
    def get_action(self, obs: np.ndarray, **kwargs) -> int:
        pass


class ActionProviderRandom(ActionProvider):
    def get_action(self, obs: np.ndarray, **kwargs) -> int:
        return random.randint(0, 4)

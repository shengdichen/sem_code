import random

import numpy as np

from src.ours.eval.common.action_provider import ActionProvider


class ActionProviderRandom(ActionProvider):
    def get_action(self, obs: np.ndarray, **kwargs) -> int:
        return random.randint(0, 4)


class ActionProviderRandomCont(ActionProvider):
    def get_action(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        return np.array(np.random.uniform([-2.5, -2.5], [+2.5, +2.5]))

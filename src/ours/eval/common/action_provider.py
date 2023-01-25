import numpy as np


class ActionProvider:
    def get_action(self, obs: np.ndarray, **kwargs) -> int | np.ndarray:
        pass

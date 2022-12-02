import numpy as np
from gym import spaces


class SpacesGenerator:
    def __init__(self, side_length: int):
        self._side_length = side_length

    def get_spaces(self):
        return self._get_observation_space(), self._get_action_space()

    def _get_observation_space(self):
        n_movements_to_observe = 4

        return spaces.Box(
            low=np.zeros(n_movements_to_observe, dtype=np.float64),
            high=np.ones(n_movements_to_observe, dtype=np.float64) * self._side_length,
            dtype=np.float64,
        )

    @staticmethod
    def _get_action_space():
        n_legal_actions = 5
        return spaces.Discrete(n_legal_actions)


class ActionConverter:
    def __init__(self, action: int, action_space: spaces.Space):
        assert action_space.contains(action), "Invalid Action"

        self._action = action

    def get_shift(self):
        if self._action == 0:
            shift = 0, 2
        elif self._action == 1:
            shift = 0, -2
        elif self._action == 2:
            shift = 2, 0
        elif self._action == 3:
            shift = -2, 0
        else:
            shift = 0, 0

        return shift

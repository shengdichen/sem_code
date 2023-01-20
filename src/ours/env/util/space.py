import numpy as np
from gym import spaces


class SpaceGeneratorBase:
    def __init__(self, side_length: int):
        self._side_length = side_length

    def get_spaces(self) -> tuple[spaces.Space, spaces.Space]:
        return self._get_observation_space(), self._get_action_space()

    def _get_observation_space(self) -> spaces.Space:
        n_movements_to_observe = 4

        return spaces.Box(
            low=np.zeros(n_movements_to_observe, dtype=np.float64),
            high=np.ones(n_movements_to_observe, dtype=np.float64) * self._side_length,
            dtype=np.float64,
        )

    def _get_action_space(self) -> spaces.Space:
        pass


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
    def __init__(self, action_raw: int, action_space: spaces.Space):
        assert action_space.contains(action_raw), "Invalid Action"

        self._action_raw = action_raw

    def get_action_converted(self):
        if self._action_raw == 0:
            shift = 0, 2
        elif self._action_raw == 1:
            shift = 0, -2
        elif self._action_raw == 2:
            shift = 2, 0
        elif self._action_raw == 3:
            shift = -2, 0
        else:
            shift = 0, 0

        return shift


class SpaceGeneratorCont:
    def __init__(self, side_length: int):
        self._side_length = side_length

    def get_spaces(self) -> tuple[spaces.Box, spaces.Box]:
        return self._get_observation_space(), self._get_action_space()

    def _get_observation_space(self) -> spaces.Box:
        n_movements_to_observe = 4

        return spaces.Box(
            low=np.zeros(n_movements_to_observe, dtype=np.float64),
            high=np.ones(n_movements_to_observe, dtype=np.float64) * self._side_length,
            dtype=np.float64,
        )

    @staticmethod
    def _get_action_space() -> spaces.Box:
        action_lower_bound, action_upper_bound = -2.5, +2.5

        return spaces.Box(
            low=np.array([action_lower_bound, action_upper_bound]),
            high=np.array([action_upper_bound, action_upper_bound]),
            dtype=np.float64,
        )


class ActionConverterCont:
    def __init__(self, action_raw: np.ndarray, action_space: spaces.Space):
        assert action_space.contains(action_raw), "Invalid Action"

        self._action_raw = action_raw

    def convert_one_dimension(self) -> np.ndarray:
        return np.round(self._action_raw)

import numpy as np
from gym import spaces


class SpacesGenerator:
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


class DiscreteSpacesGenerator(SpacesGenerator):
    def __init__(self, side_length: int):
        super().__init__(side_length)

    def _get_action_space(self) -> spaces.Space:
        n_legal_actions = 5
        return spaces.Discrete(n_legal_actions)


class ContSpacesGenerator(SpacesGenerator):
    def __init__(self, side_length: int):
        super().__init__(side_length)

    def _get_action_space(self) -> spaces.Box:
        action_lower_bound, action_upper_bound = -2.5, +2.5

        return spaces.Box(
            low=np.array([action_lower_bound, action_lower_bound]),
            high=np.array([action_upper_bound, action_upper_bound]),
            dtype=np.float64,
        )

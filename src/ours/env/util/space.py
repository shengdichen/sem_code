import numpy as np
from gym import spaces


class SpacesGeneratorBase:
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


class SpacesGenerator(SpacesGeneratorBase):
    def __init__(self, side_length: int):
        super().__init__(side_length)

    def _get_action_space(self) -> spaces.Space:
        n_legal_actions = 5
        return spaces.Discrete(n_legal_actions)


class ActionConverterBase:
    def __init__(self, action_raw: int | np.ndarray, action_space: spaces.Space):
        assert action_space.contains(action_raw), "Invalid Action"

        self._action_raw = action_raw

    def get_action_converted(self) -> tuple[int, int]:
        pass


class ActionConverter(ActionConverterBase):
    def __init__(self, action_raw: int, action_space: spaces.Space):
        super().__init__(action_raw, action_space)

    def get_action_converted(self) -> tuple[int, int]:
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


class SpacesGeneratorCont(SpacesGeneratorBase):
    def __init__(self, side_length: int):
        super().__init__(side_length)

    def _get_action_space(self) -> spaces.Box:
        action_lower_bound, action_upper_bound = -2.5, +2.5

        return spaces.Box(
            low=np.array([action_lower_bound, action_lower_bound]),
            high=np.array([action_upper_bound, action_upper_bound]),
            dtype=np.float64,
        )


class ActionConverterCont(ActionConverterBase):
    def __init__(self, action_raw: np.ndarray, action_space: spaces.Space):
        super().__init__(action_raw, action_space)

    def get_action_converted(self) -> tuple[int, int]:
        shift_x, shift_y = np.round(self._action_raw)
        return int(shift_x), int(shift_y)

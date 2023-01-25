import numpy as np
from gym import spaces


class ActionConverter:
    def __init__(self, action_raw: int | np.ndarray, action_space: spaces.Space):
        assert action_space.contains(action_raw), "Invalid Action"

        self._action_raw = action_raw

    def get_action_converted(self) -> tuple[int, int]:
        pass


class DiscreteActionConverter(ActionConverter):
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


class ContActionConverter(ActionConverter):
    def __init__(self, action_raw: np.ndarray, action_space: spaces.Space):
        super().__init__(action_raw, action_space)

    def get_action_converted(self) -> tuple[int, int]:
        shift_x, shift_y = np.round(self._action_raw)
        return int(shift_x), int(shift_y)

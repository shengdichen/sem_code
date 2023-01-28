import numpy as np

from src.ours.rl.pwil.param import PwilParam


class Selector:
    def __init__(self, trajectories: list[np.ndarray], pwil_params: list[PwilParam]):
        self._trajectories = trajectories
        self._params = pwil_params

    @property
    def trajectories(self) -> list[np.ndarray]:
        return self._trajectories

    @property
    def params(self) -> list[PwilParam]:
        return self._params

    def select_by_trajectory_num(self, candidates: list[int]) -> "Selector":
        self._trajectories, self._params = self._select_by(
            self._is_selected_by_trajectory_num, candidates
        )
        return self

    def select_by_n_demos(self, candidates: list[int]) -> "Selector":
        self._trajectories, self._params = self._select_by(
            self._is_selected_by_n_demos, candidates
        )
        return self

    def select_by_subsampling(self, candidates: list[int]) -> "Selector":
        self._trajectories, self._params = self._select_by(
            self._is_selected_by_subsampling, candidates
        )
        return self

    def _select_by(
        self, _is_selected_function, candidates: list[int]
    ) -> tuple[list[np.ndarray], list[PwilParam]]:
        selected_trajectories, selected_params = [], []
        for trajectory, param in zip(self._trajectories, self._params):
            if _is_selected_function(param, candidates):
                selected_trajectories.append(trajectory)
                selected_params.append(param)

        return selected_trajectories, selected_params

    @staticmethod
    def _is_selected_by_trajectory_num(param: PwilParam, candidates: list[int]) -> bool:
        for candidate in candidates:
            if param.trajectory_num == candidate:
                return True
        return False

    @staticmethod
    def _is_selected_by_n_demos(param: PwilParam, candidates: list[int]) -> bool:
        for candidate in candidates:
            if param.pwil_training_param["n_demos"] == candidate:
                return True
        return False

    @staticmethod
    def _is_selected_by_subsampling(param: PwilParam, candidates: list[int]) -> bool:
        for candidate in candidates:
            if param.pwil_training_param["subsampling"] == candidate:
                return True
        return False

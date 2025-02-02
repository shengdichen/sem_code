import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.rl.expert.sb3.manager import ExpertSb3Manager
from src.ours.rl.expert.trajectory import ExpertTrajectoryManager


class ExpertManager:
    def __init__(
        self,
        managers: tuple[ExpertSb3Manager, ExpertTrajectoryManager],
        env_identifier: str,
    ):
        self._sb3_manager, self._trajectory_manager = managers
        self._env_identifier = env_identifier

    def train_and_save_model(self) -> None:
        self._sb3_manager.train()
        self._sb3_manager.save()

    def save_trajectory(self) -> None:
        self._trajectory_manager.save()

    def load_model(self) -> BaseAlgorithm:
        return self._sb3_manager.model

    def load_trajectory(self) -> np.ndarray:
        return self._trajectory_manager.load()

    def save_trajectory_stats(self) -> None:
        self._trajectory_manager.save_stats()

    def show_trajectory_stats(self) -> None:
        self._trajectory_manager.show_stats()

    def save_trajectory_plot(self) -> None:
        self._trajectory_manager.save_plot()

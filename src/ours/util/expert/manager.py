import numpy as np

from src.ours.util.expert.sb3.manager import Sb3Manager
from src.ours.util.expert.trajectory.manager import TrajectoryManager


class ExpertManager:
    def __init__(
        self,
        managers: tuple[Sb3Manager, TrajectoryManager],
        env_identifier: str,
    ):
        self._sb3_manager, self._trajectory_manager = managers
        self._env_identifier = env_identifier

    def train_model(self) -> None:
        self._sb3_manager.train()

    def save_model_and_trajectory(self) -> None:
        self._sb3_manager.save()
        self._trajectory_manager.save()

    def load_trajectory(self) -> np.ndarray:
        return self._trajectory_manager.load()

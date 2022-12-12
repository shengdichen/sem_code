import numpy as np

from src.ours.util.expert.sb3.manager import Sb3Manager
from src.ours.util.expert.trajectory.manager import TrajectoryManager
from src.ours.util.expert.sb3.util.train import TrainerExpert


class ClientExpert:
    def __init__(
        self,
        trainer: TrainerExpert,
        managers: tuple[Sb3Manager, TrajectoryManager],
        env_identifier: str,
    ):
        self._trainer = trainer
        self._saver_manager, self._trajectory_manager = managers
        self._env_identifier = env_identifier

    def train(self) -> None:
        self._trainer.train()

    def save(self) -> None:
        self._saver_manager.save(self._env_identifier)
        self._trajectory_manager.save_trajectory(self._env_identifier)

    def load(self) -> np.ndarray:
        return self._trajectory_manager.load_trajectory(self._env_identifier)

import numpy as np

from src.ours.rl.common.trajectory.analyzer.stats.single import TrajectoryStats


class TrajectoriesStats:
    def __init__(self, trajectories: list[np.ndarray]):
        self._trajectories = trajectories

        self._trajectories_stats = [
            TrajectoryStats(trajectory) for trajectory in self._trajectories
        ]

    def show_stats(self) -> None:
        for trajectory_stats in self._trajectories_stats:
            print(trajectory_stats.get_stats())

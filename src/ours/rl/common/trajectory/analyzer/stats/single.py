import numpy as np

from src.ours.rl.common.trajectory.analyzer.info import TrajectoryInfo


class TrajectoryStats:
    def __init__(self, trajectory: np.ndarray):
        self._trajectory = trajectory

        self._info = TrajectoryInfo(trajectory)

        self._rewards_avg_std, self._rewards_min_max = (
            AvgStdUtil(self._info.rewards_per_episode),
            MinMaxUtil(self._info.rewards_per_episode),
        )
        self._lengths_avg_std, self._lengths_min_max = (
            AvgStdUtil(self._info.lengths_per_episode),
            MinMaxUtil(self._info.lengths_per_episode),
        )

    def get_stats(self) -> str:
        stats = ""
        stats += "{0:*^60}\n".format(" Trajectory Statistics [START] ")

        stats += "Number of steps: {0}\n".format(self._info.n_steps)
        stats += "Reward (per step): {0}\n".format(AvgStdUtil(self._info.reward))
        stats += "Reward (per step): {0}\n".format(MinMaxUtil(self._info.reward))
        stats += "\n"

        stats += "Number of episodes: {0}\n".format(self._info.n_episodes)
        stats += "Reward (per episode): {0}\n".format(self._rewards_avg_std)
        stats += "Reward (per episode): {0}\n".format(self._rewards_min_max)
        stats += "Length (per episode): {0}\n".format(self._lengths_avg_std)
        stats += "Length (per episode): {0}\n".format(self._lengths_min_max)

        stats += "{0:*^60}\n".format(" Trajectory Statistics [END] ")

        return stats


class AvgStdUtil:
    def __init__(self, data: np.ndarray):
        self._data = data
        self._stats = self._make_stats()

    @property
    def stats(self) -> tuple[float, float]:
        return self._stats

    def _make_stats(self) -> tuple[float, float]:
        return float(np.mean(self._data)), float(np.std(self._data))

    def __str__(self):
        return "[<avg> +- <std>] | {0:.4} +- {1:.4}".format(
            self._stats[0], self._stats[1]
        )


class MinMaxUtil:
    def __init__(self, data: np.ndarray):
        self._data = data
        self._stats = self._make_stats()

    @property
    def stats(self) -> tuple[float, float]:
        return self._stats

    def _make_stats(self) -> tuple[float, float]:
        return float(np.min(self._data)), float(np.max(self._data))

    def __str__(self):
        return "[<min> / <max>] | {0:.4} / {1:.4}".format(
            self._stats[0], self._stats[1]
        )

    @staticmethod
    def get_np_min_max(vec: np.ndarray) -> tuple[float, float]:
        return float(np.min(vec)), float(np.max(vec))

    @staticmethod
    def get_np_min_max_x_y(
        vec_x: np.ndarray, vec_y: np.ndarray
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        return MinMaxUtil.get_np_min_max(vec_x), MinMaxUtil.get_np_min_max(vec_y)

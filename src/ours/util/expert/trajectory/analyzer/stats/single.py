import numpy as np


class TrajectoryInfo:
    def __init__(self, trajectory: np.ndarray):
        self._trajectory = trajectory

        (
            self._rewards_per_episode,
            self._lengths_per_episode,
        ) = self._get_rewards_and_lengths_per_episode()

    def _get_rewards_and_lengths_per_episode(self) -> tuple[np.ndarray, np.ndarray]:
        rewards_per_episode, lengths_per_episode = [], []
        reward_current_episode, length_current_episode = 0, 0

        for data_current_step in self._trajectory:
            reward_current_episode += data_current_step[-2]
            length_current_episode += 1
            if data_current_step[-1]:  # current episode is over at this step
                rewards_per_episode.append(reward_current_episode)
                lengths_per_episode.append(length_current_episode)
                reward_current_episode, length_current_episode = 0, 0

        return np.array(rewards_per_episode), np.array(lengths_per_episode)

    @property
    def agent_pos(self) -> tuple[np.ndarray, np.ndarray]:
        return self._trajectory[:, 0], self._trajectory[:, 1]

    @property
    def target_pos(self) -> tuple[np.ndarray, np.ndarray]:
        return self._trajectory[:, 2], self._trajectory[:, 3]

    @property
    def action(self) -> np.ndarray:
        return self._trajectory[:, 4:-2]

    @property
    def reward(self) -> np.ndarray:
        return self._trajectory[:, -2]

    @property
    def rewards_per_episode(self) -> np.ndarray:
        return self._rewards_per_episode

    @property
    def lengths_per_episode(self) -> np.ndarray:
        return self._lengths_per_episode

    @property
    def done(self) -> np.ndarray:
        return self._trajectory[:, -1]

    @property
    def n_episodes(self):
        return int(np.sum(self.done))


class TrajectoryStats:
    def __init__(self, trajectory: np.ndarray):
        self._trajectory = trajectory

        self._info = TrajectoryInfo(trajectory)

    def get_stats(self) -> str:
        stats = ""
        stats += "{0:*^60}\n".format(" Trajectory Statistics [START] ")

        stats += "Reward (global): {0}\n".format(AvgStdUtil(self._info.reward))
        stats += "Reward (global): {0}\n".format(MinMaxUtil(self._info.reward))

        stats += "Number of episodes: {0}\n".format(self._info.n_episodes)
        stats += "Reward (episode): {0}\n".format(
            AvgStdUtil(self._info.rewards_per_episode)
        )
        stats += "Reward (episode): {0}\n".format(
            MinMaxUtil(self._info.rewards_per_episode)
        )
        stats += "Length (episode): {0}\n".format(
            AvgStdUtil(self._info.lengths_per_episode)
        )

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

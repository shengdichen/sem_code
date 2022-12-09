import numpy as np


class TrajectoryInterpreter:
    def __init__(self, trajectory: np.ndarray):
        self._trajectory = trajectory

        self._rewards_per_episode = self._get_rewards_per_episode()

    def _get_rewards_per_episode(self) -> np.ndarray:
        rewards_per_episode = []
        reward_current_episode = 0

        for data_current_step in self._trajectory:
            reward_current_episode += data_current_step[-2]
            if data_current_step[-1]:  # current episode is over at this step
                rewards_per_episode.append(reward_current_episode)
                reward_current_episode = 0

        return np.array(rewards_per_episode)

    @property
    def agent_pos(self) -> tuple[np.ndarray, np.ndarray]:
        return self._trajectory[:, 0], self._trajectory[:, 1]

    @property
    def target_pos(self) -> tuple[np.ndarray, np.ndarray]:
        return self._trajectory[:, 2], self._trajectory[:, 3]

    @property
    def action(self) -> np.ndarray:
        return self._trajectory[:, 4]

    @property
    def reward(self) -> np.ndarray:
        return self._trajectory[:, 5]

    @property
    def done(self) -> np.ndarray:
        return self._trajectory[:, 6]

    def display_stats(self) -> None:
        print("{0:*^60}".format(" Trajectory Statistics [START] "))

        print("Number of episodes: {0}".format(self._get_num_episodes()))

        print("Reward (global): {0}".format(AvgStdUtil(self.reward)))
        print("Reward (global): {0}".format(MinMaxUtil(self.reward)))

        print("Reward (episode): {0}".format(AvgStdUtil(self._rewards_per_episode)))
        print("Reward (episode): {0}".format(MinMaxUtil(self._rewards_per_episode)))

        print("{0:*^60}".format(" Trajectory Statistics [END] "))
        print()

    def _get_num_episodes(self) -> int:
        return int(np.sum(self.done))


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

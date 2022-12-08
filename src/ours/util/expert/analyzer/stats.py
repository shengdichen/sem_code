import numpy as np


class TrajectoryInterpreter:
    def __init__(self, trajectory: np.ndarray):
        self._trajectory = trajectory

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
        num_episodes = self._get_num_episodes()

        (
            ep_rew_avg,
            ep_rew_std,
            ep_rew_min,
            ep_rew_max,
        ) = self._get_episode_reward_stats()

        print("Demo file stats")
        print("-------------")
        print("Number of episodes: ", num_episodes)
        print("Reward stats: {0}".format(AvgStdUtil(self.reward)))
        print("Reward min / max {0}".format(MinMaxUtil(self.reward)))
        print("Episode reward stats: ", ep_rew_avg, " +- ", ep_rew_std)
        print("Episode reward min / max", ep_rew_min, " / ", ep_rew_max)
        print("-------------")

    def _get_num_episodes(self) -> int:
        return int(np.sum(self.done))

    def _get_reward_stats(self) -> tuple[float, float, float, float]:
        # reward stats
        rew_avg, rew_std = TrajectoryInterpreter._get_avg_std(self.reward)
        rew_min, rew_max = MinMaxUtil.get_np_min_max(self.reward)

        return rew_avg, rew_std, rew_min, rew_max

    def _get_episode_reward_list(self) -> np.ndarray:
        ep_rew_list = []
        ep_rew = 0
        for sard in self._trajectory:
            ep_rew += sard[-2]
            if sard[-1] == 1:
                ep_rew_list.append(ep_rew)
                # print("episode_reward", ep_rew)
                ep_rew = 0

        return np.array(ep_rew_list)

    def _get_episode_reward_stats(self) -> tuple[float, float, float, float]:
        ep_rew_list = self._get_episode_reward_list()

        ep_rew_avg, ep_rew_std = TrajectoryInterpreter._get_avg_std(ep_rew_list)
        ep_rew_min, ep_rew_max = MinMaxUtil.get_np_min_max(ep_rew_list)

        return ep_rew_avg, ep_rew_std, ep_rew_min, ep_rew_max

    @staticmethod
    def _get_avg_std(data: np.ndarray) -> tuple[float, float]:
        return float(np.mean(data)), float(np.std(data))


class MinMaxUtil:
    def __init__(self, data: np.ndarray):
        self._data = data
        self._stats = self._make_stats()

    @property
    def stats(self):
        return self._stats

    def _make_stats(self):
        return float(np.min(self._data)), float(np.max(self._data))

    def __str__(self):
        return "[<min> / <max>] | {0} / {1}".format(self._stats[0], self._stats[1])

    @staticmethod
    def get_np_min_max(vec: np.ndarray) -> tuple[float, float]:
        return float(np.min(vec)), float(np.max(vec))

    @staticmethod
    def get_np_min_max_x_y(vec_x: np.ndarray, vec_y: np.ndarray):
        return MinMaxUtil.get_np_min_max(vec_x), MinMaxUtil.get_np_min_max(vec_y)


class AvgStdUtil:
    def __init__(self, data: np.ndarray):
        self._data = data
        self._stats = self._make_stats()

    @property
    def stats(self):
        return self._stats

    def _make_stats(self):
        return float(np.mean(self._data)), float(np.std(self._data))

    def __str__(self):
        return "[<avg> +- <std>] | {0} +- {1}".format(self._stats[0], self._stats[1])

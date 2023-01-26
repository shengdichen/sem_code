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
    def n_steps(self):
        return self._trajectory.shape[0]

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

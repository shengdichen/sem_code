from typing import Any

import numpy as np
from gym import Env
from stable_baselines3.common.base_class import BaseAlgorithm


class TrajectoryGeneratorConfig:
    def __init__(self):
        self._nr_trajectories = 10

        self._render = False
        self._deterministic = False

    @property
    def nr_trajectories(self):
        return self._nr_trajectories

    @property
    def render(self):
        return self._render

    @property
    def deterministic(self):
        return self._deterministic


class TrajectoryGenerator:
    def __init__(
        self,
        env_model: tuple[Env, BaseAlgorithm | Any],
        trajectory_generator_config=TrajectoryGeneratorConfig(),
    ):
        self._env, self._model = env_model
        self._trajectory_generator_config = trajectory_generator_config

        self._trajectory, self._num_steps = [], 0

    def get_trajectory(self) -> np.ndarray:
        for __ in range(self._trajectory_generator_config.nr_trajectories):
            obs_curr = self._env.reset()
            done = False
            reward_curr_episode, snapshots_curr_episode = 0, []

            while not done:
                action, __ = self._model.predict(
                    obs_curr,
                    deterministic=self._trajectory_generator_config.deterministic,
                )
                obs_next, reward, done, __ = self._env.step(action)
                if self._trajectory_generator_config.render:
                    self._env.render()

                snapshot_curr_step = np.hstack(
                    [np.squeeze(obs_next), np.squeeze(action), reward, done]
                )
                self._trajectory.append(snapshot_curr_step)
                snapshots_curr_episode.append(snapshot_curr_step)

                reward_curr_episode += reward
                self._num_steps += 1
                obs_curr = obs_next

            print("Episode reward: ", reward_curr_episode)

        self._env.close()

        return np.stack(self._trajectory)

    def _append_one_trajectory(self) -> None:
        obs_curr = self._env.reset()
        done = False
        reward_curr_episode, snapshots_curr_episode = 0, []

        while not done:
            action, __ = self._model.predict(
                obs_curr,
                deterministic=self._trajectory_generator_config.deterministic,
            )
            obs_next, reward, done, __ = self._env.step(action)
            if self._trajectory_generator_config.render:
                self._env.render()

            snapshot_curr_step = np.hstack(
                [np.squeeze(obs_next), np.squeeze(action), reward, done]
            )
            self._trajectory.append(snapshot_curr_step)
            snapshots_curr_episode.append(snapshot_curr_step)

            reward_curr_episode += reward
            self._num_steps += 1
            obs_curr = obs_next

        print("Episode reward: ", reward_curr_episode)

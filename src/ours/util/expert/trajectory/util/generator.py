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

    def get_trajectory(self) -> np.ndarray:
        num_steps = 0
        trajectory = []

        for __ in range(self._trajectory_generator_config.nr_trajectories):
            obs = self._env.reset()
            done = False
            total_reward = 0
            snapshots_curr_episode = []

            while not done:
                action, _states = self._model.predict(
                    obs, deterministic=self._trajectory_generator_config.deterministic
                )
                next_obs, reward, done, _ = self._env.step(action)

                obs = next_obs
                total_reward += reward
                snapshot_curr_step = np.hstack(
                    [np.squeeze(obs), np.squeeze(action), reward, done]
                )

                trajectory.append(snapshot_curr_step)
                snapshots_curr_episode.append(snapshot_curr_step)
                num_steps += 1

                if self._trajectory_generator_config.render:
                    self._env.render()

            print("Episode reward: ", total_reward)

        self._env.close()

        return np.stack(trajectory)

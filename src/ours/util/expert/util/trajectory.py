from typing import Any

import numpy as np
from gym import Env


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
        env_model: tuple[Env, Any],
        expert_manager_param=TrajectoryGeneratorConfig(),
    ):
        self._env, self._model = env_model
        self._expert_manager_param = expert_manager_param

    def get_trajectory(self) -> np.ndarray:
        num_steps = 0
        trajectories_expert = []

        for __ in range(self._expert_manager_param.nr_trajectories):
            obs = self._env.reset()
            done = False
            total_reward = 0
            trajectories_episode = []

            while not done:
                action, _states = self._model.predict(
                    obs, deterministic=self._expert_manager_param.deterministic
                )
                next_obs, reward, done, _ = self._env.step(action)

                obs = next_obs
                total_reward += reward
                trajectory_curr_step = np.hstack(
                    [np.squeeze(obs), np.squeeze(action), reward, done]
                )
                trajectories_expert.append(trajectory_curr_step)
                trajectories_episode.append(trajectory_curr_step)
                num_steps += 1
                if self._expert_manager_param.render:
                    self._env.render()

            print("Episode reward: ", total_reward)

        self._env.close()

        return np.stack(trajectories_expert)

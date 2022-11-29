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
        expert_traj = []

        for i_episode in range(self._expert_manager_param.nr_trajectories):
            ob = self._env.reset()
            done = False
            total_reward = 0
            episode_traj = []

            while not done:
                ac, _states = self._model.predict(
                    ob, deterministic=self._expert_manager_param.deterministic
                )
                next_ob, reward, done, _ = self._env.step(ac)

                ob = next_ob
                total_reward += reward
                stacked_vec = np.hstack([np.squeeze(ob), np.squeeze(ac), reward, done])
                expert_traj.append(stacked_vec)
                episode_traj.append(stacked_vec)
                num_steps += 1
                if self._expert_manager_param.render:
                    self._env.render()

            print("Episode reward: ", total_reward)

        self._env.close()

        return np.stack(expert_traj)

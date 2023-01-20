import numpy as np

from src.ours.env.env import MovePoint, MovePointBase
from src.ours.eval.pointenv.run.actionprovider import (
    ActionProvider,
    ActionProviderRandom,
)


class PointEnvRunnerConfig:
    n_max_steps_per_episode = 500
    n_episodes = 2


class PointEnvRunnerBase:
    def __init__(self, env: MovePointBase):
        self._env = env
        self._n_max_steps_per_episode, self._n_episodes = 500, 1

        self._obs, self._done = self.reset()

    def reset(self) -> tuple[np.ndarray, bool]:
        obs = self._env.reset()
        self._env.render("human")
        return obs, False

    def run_episodes(self, action_provider: ActionProvider) -> None:
        for __ in range(PointEnvRunnerConfig.n_episodes):
            self.reset()
            self._run_one_episode(action_provider)

        self._env.close()

    def _run_one_episode(self, action_provider: ActionProvider) -> None:
        for __ in range(PointEnvRunnerConfig.n_max_steps_per_episode):
            self._obs, __, self._done, __ = self._env.step(
                action_provider.get_action(self._obs)
            )
            self._env.render("human")

            if self._done:
                break


class PointEnvRunner:
    def __init__(self):
        self._env = MovePoint()
        self._n_max_steps_per_episode, self._n_episodes = 500, 1

        self._obs, self._done = self.reset()

    def reset(self) -> tuple[np.ndarray, bool]:
        obs = self._env.reset()
        self._env.render("human")
        return obs, False

    def run_episodes(self, action_provider: ActionProvider) -> None:
        for __ in range(PointEnvRunnerConfig.n_episodes):
            self.reset()
            self._run_one_episode(action_provider)

        self._env.close()

    def _run_one_episode(self, action_provider: ActionProvider) -> None:
        for __ in range(PointEnvRunnerConfig.n_max_steps_per_episode):
            self._obs, __, self._done, __ = self._env.step(
                action_provider.get_action(self._obs)
            )
            self._env.render("human")

            if self._done:
                break


def client_code():
    runner = PointEnvRunner()
    runner.run_episodes(ActionProviderRandom())


if __name__ == "__main__":
    client_code()

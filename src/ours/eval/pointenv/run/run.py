import numpy as np

from src.ours.env.env import MovePoint
from src.ours.eval.pointenv.run.actionprovider import (
    ActionProvider,
    ActionProviderRandom,
)


class PointEnvRunner:
    def __init__(self):
        self._env = MovePoint()
        self._n_max_steps_per_episode, self._n_episodes = 500, 1

        self._obs, self._done = self.reset()

    def reset(self) -> tuple[np.ndarray, bool]:
        obs = self._env.reset()
        self._env.render("human")
        return obs, False

    def close(self) -> None:
        self._env.close()

    def _run_one_episode(self, action_provider: "ActionProvider") -> None:
        for __ in range(self._n_max_steps_per_episode):
            self._obs, __, self._done, __ = self._env.step(
                action_provider.get_action(self._obs)
            )
            self._env.render()

            if self._done:
                break

    def run_episodes(self, action_provider: "ActionProvider" = None) -> None:
        if action_provider is None:
            action_provider = ActionProviderRandom()

        for __ in range(self._n_episodes):
            self.reset()
            self._run_one_episode(action_provider)

        self._env.close()


def client_code():
    runner = PointEnvRunner()
    runner.run_episodes()


if __name__ == "__main__":
    client_code()

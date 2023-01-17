import random

import numpy as np

from src.ours.env.env import MovePoint


class PointEnvRunner:
    def __init__(self):
        self._env = MovePoint()
        self._n_steps = 500

        self._obs, self._done = self.reset()

    def reset(self) -> tuple[np.ndarray, bool]:
        obs = self._env.reset()
        self._env.render("human")
        return obs, False

    def close(self) -> None:
        self._env.close()

    def run_one_episode(self, action_provider: "ActionProvider") -> None:
        for __ in range(self._n_steps):
            self._obs, __, self._done, __ = self._env.step(
                action_provider.get_action(self._obs)
            )
            self._env.render()

    def run_random(
        self, n_runs: int = 1, action_provider: "ActionProvider" = None
    ) -> None:
        if action_provider is None:
            action_provider = ActionProviderRandom()

        for __ in range(n_runs):
            self.reset()
            self.run_one_episode(action_provider)

        self._env.close()


class ActionProvider:
    def get_action(self, obs: np.ndarray, **kwargs) -> int:
        pass


class ActionProviderRandom(ActionProvider):
    def get_action(self, obs: np.ndarray, **kwargs) -> int:
        return random.randint(0, 4)


def client_code():
    runner = PointEnvRunner()
    runner.run_random()


if __name__ == "__main__":
    client_code()

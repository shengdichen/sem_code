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

    def take_action_random(self) -> None:
        for __ in range(self._n_steps):
            self._env.step(random.randint(0, 4))
            self._env.render()

    def run_random(self, n_runs: int = 1) -> None:
        for __ in range(n_runs):
            self.reset()
            self.take_action_random()

        self._env.close()


def client_code():
    runner = PointEnvRunner()
    runner.run_random()


if __name__ == "__main__":
    client_code()

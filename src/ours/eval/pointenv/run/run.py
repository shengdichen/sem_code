import numpy as np

from src.ours.env.env import DiscretePointNav, PointNav, ContPointNav
from src.ours.eval.common.action_provider import ActionProvider


class PointEnvRunnerConfig:
    n_max_steps_per_episode = 500
    n_episodes = 2


class PointEnvRunner:
    def __init__(self, env: PointNav):
        self._env = env

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


class DiscretePointEnvRunner(PointEnvRunner):
    def __init__(self):
        super().__init__(DiscretePointNav())


class ContPointEnvRunner(PointEnvRunner):
    def __init__(self):
        super().__init__(ContPointNav())


def client_code():
    from src.ours.eval.pointenv.run.action_provider import (
        ActionProviderRandom,
        ActionProviderRandomCont,
    )

    runner = DiscretePointEnvRunner()
    runner.run_episodes(ActionProviderRandom())

    runner = ContPointEnvRunner()
    runner.run_episodes(ActionProviderRandomCont())


if __name__ == "__main__":
    client_code()

import numpy as np

from src.ours.env.creation import (
    PointEnvFactory,
    PointEnvIdentifierGenerator,
    PointEnvConfigFactory,
)
from src.ours.util.common.param import ExpertParam
from src.ours.util.expert.sb3.manager import Sb3Manager
from src.ours.util.expert.client import ClientExpert
from src.ours.util.expert.manager import ExpertManager
from src.ours.util.expert.sb3.util.train import TrainerExpert
from src.ours.util.common.helper import Plotter


class PointEnvExpertSingle:
    def __init__(self, env_config: dict[str:int]):
        self._training_param = ExpertParam()

        env = PointEnvFactory(env_config).create()
        trainer = TrainerExpert(env, self._training_param)
        env_identifier = PointEnvIdentifierGenerator().from_env(env)

        self._expert_client = ClientExpert(
            trainer,
            (
                Sb3Manager(trainer.model, self._training_param),
                ExpertManager((env, trainer.model), self._training_param),
            ),
            env_identifier,
        )

    def train_and_save(self) -> None:
        self._expert_client.train()
        self._expert_client.save()

    def load(self) -> np.ndarray:
        return self._expert_client.load()


class PointEnvExpert:
    def __init__(self):
        self._training_param = ExpertParam()
        self._n_timesteps = self._training_param.n_steps_expert_train

        self._env_configs = PointEnvConfigFactory().env_configs

    def train_and_plot(self) -> None:
        self._train_default_configs()
        self._plot()

    def _train_default_configs(self):
        """
        # Train experts with different shifts representing their waypoint preferences
        """

        for env_config in self._env_configs:
            self._train_and_save(env_config)

    def _train_and_save(self, env_config: dict[str:int]) -> None:
        env = PointEnvFactory(env_config).create()
        trainer = TrainerExpert(env, self._training_param)
        env_identifier = PointEnvIdentifierGenerator().from_env(env)

        expert_client = ClientExpert(
            trainer,
            (
                Sb3Manager(trainer.model, self._training_param),
                ExpertManager((env, trainer.model), self._training_param),
            ),
            env_identifier,
        )
        expert_client.train()
        expert_client.save()

    def _plot(self) -> None:
        Plotter.plot_experts(self._n_timesteps)
        Plotter.plot_experts(self._n_timesteps, hist=False)


def client_code():
    trainer = PointEnvExpert()
    trainer.train_and_plot()


if __name__ == "__main__":
    client_code()

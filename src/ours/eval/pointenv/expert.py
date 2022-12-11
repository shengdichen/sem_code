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
from src.ours.util.expert.analyzer.general import TrajectoriesAnalyzer


class PointEnvExpertSingle:
    def __init__(self, training_param: ExpertParam, env_config: dict[str:int]):
        self._training_param = training_param

        self._expert_client = self._make_expert_client(env_config)

    def _make_expert_client(self, env_config: dict[str:int]) -> ClientExpert:
        env = PointEnvFactory(env_config).create()
        trainer = TrainerExpert(env, self._training_param)
        env_identifier = PointEnvIdentifierGenerator().from_env(env)

        return ClientExpert(
            trainer,
            (
                Sb3Manager(trainer.model, self._training_param),
                ExpertManager((env, trainer.model), self._training_param),
            ),
            env_identifier,
        )

    def train(self) -> None:
        self._expert_client.train()

    def save(self) -> None:
        self._expert_client.save()

    def load(self) -> np.ndarray:
        return self._expert_client.load()


class PointEnvExpertDefault:
    def __init__(self):
        self._training_param = ExpertParam()
        self._n_timesteps = self._training_param.n_steps_expert_train

        self._env_configs = PointEnvConfigFactory().env_configs
        self._pointenv_experts = self._make_pointenv_experts()

    def _make_pointenv_experts(self) -> list[PointEnvExpertSingle]:
        pointenv_experts = []
        for env_config in self._env_configs:
            pointenv_experts.append(
                PointEnvExpertSingle(self._training_param, env_config)
            )

        return pointenv_experts

    def train_and_plot(self) -> None:
        self.train_and_save_all()
        self.plot()

    def train_and_save_all(self) -> None:
        for pointenv_expert in self._pointenv_experts:
            pointenv_expert.train()
            pointenv_expert.save()

    def plot(self) -> None:
        TrajectoriesAnalyzer(self.load_all()).analyze()

    def load_all(self) -> list[np.ndarray]:
        expert_demos = []
        for pointenv_expert in self._pointenv_experts:
            demo = pointenv_expert.load()
            expert_demos.append(demo)

        return expert_demos


def client_code():
    trainer = PointEnvExpertDefault()
    trainer.train_and_plot()


if __name__ == "__main__":
    client_code()

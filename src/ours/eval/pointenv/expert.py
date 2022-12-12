import numpy as np

from src.ours.env.creation import (
    PointEnvFactory,
    PointEnvIdentifierGenerator,
    PointEnvConfigFactory,
)
from src.ours.util.common.param import ExpertParam
from src.ours.util.expert.manager import ExpertManager
from src.ours.util.expert.sb3.manager import Sb3Manager
from src.ours.util.expert.trajectory.analyzer.general import TrajectoriesAnalyzer
from src.ours.util.expert.trajectory.manager import TrajectoryManager


class PointEnvExpertManagerFactory:
    def __init__(self, training_param: ExpertParam, env_config: dict[str:int]):
        self._training_param = training_param
        self._env_config = env_config

        self._expert_client = self.create()

    def create(self) -> ExpertManager:
        env = PointEnvFactory(self._env_config).create()
        env_identifier = PointEnvIdentifierGenerator().from_env(env)

        sb3_manager = Sb3Manager((env, env_identifier), self._training_param)
        trajectory_manager = TrajectoryManager(
            (env, env_identifier),
            (sb3_manager.model, self._training_param),
        )

        return ExpertManager(
            (sb3_manager, trajectory_manager),
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

    def _make_pointenv_experts(self) -> list[PointEnvExpertManagerFactory]:
        pointenv_experts = [
            PointEnvExpertManagerFactory(self._training_param, env_config)
            for env_config in self._env_configs
        ]

        return pointenv_experts

    def train_and_analyze(self) -> None:
        self.train_and_save()
        self.analyze()

    def train_and_save(self) -> None:
        for pointenv_expert in self._pointenv_experts:
            pointenv_expert.train()
            pointenv_expert.save()

    def analyze(self) -> None:
        TrajectoriesAnalyzer(self._load()).analyze()

    def _load(self) -> list[np.ndarray]:
        trajectories = [
            pointenv_expert.load() for pointenv_expert in self._pointenv_experts
        ]

        return trajectories


def client_code():
    trainer = PointEnvExpertDefault()
    trainer.analyze()


if __name__ == "__main__":
    client_code()

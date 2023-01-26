import numpy as np
from gym import Env

from src.ours.env.creation import (
    DiscretePointEnvFactory,
    ContPointEnvFactory,
    PointEnvFactory,
)
from src.ours.env.config import PointEnvConfigFactory
from src.ours.env.identifier import (
    PointEnvIdentifierGenerator,
    DiscretePointEnvIdentifierGenerator,
    ContPointEnvIdentifierGenerator,
)
from src.ours.eval.pointenv.run.actionprovider import ActionProvider
from src.ours.eval.pointenv.run.run import DiscretePointEnvRunner, ContPointEnvRunner
from src.ours.util.expert.param import ExpertParam
from src.ours.util.expert.manager import ExpertManager
from src.ours.util.expert.sb3.manager import ExpertSb3Manager
from src.ours.util.common.trajectory.analyzer.plot.multi import (
    ParallelTrajectoriesPlot,
    SeparateTrajectoriesPlot,
)
from src.ours.util.common.trajectory.analyzer.stats.multi import TrajectoriesStats
from src.ours.util.expert.trajectory import ExpertTrajectoryManager


class PointEnvExpertManagerFactory:
    def __init__(
        self,
        training_param: ExpertParam,
        env_config: dict[str:int],
        env_factory: PointEnvFactory,
        env_identifier_generator: PointEnvIdentifierGenerator,
    ):
        self._training_param = training_param
        self._env_config = env_config

        self._env_factory = env_factory
        self._env_identifier_generator = env_identifier_generator

    def create(self) -> ExpertManager:
        (env, env_eval), env_identifier = self._get_envs_and_identifier()

        sb3_manager = ExpertSb3Manager(
            ((env, env_eval), env_identifier), self._training_param
        )
        trajectory_manager = ExpertTrajectoryManager(
            (env, env_identifier),
            (sb3_manager.model, self._training_param),
        )

        return ExpertManager(
            (sb3_manager, trajectory_manager),
            env_identifier,
        )

    def _get_envs_and_identifier(self) -> tuple[tuple[Env, Env], str]:
        env, env_eval = (
            self._env_factory.create(),
            self._env_factory.create(),
        )
        env_identifier = self._env_identifier_generator.from_env(env)
        return (env, env_eval), env_identifier


class DiscretePointEnvExpertManagerFactory(PointEnvExpertManagerFactory):
    def __init__(self, training_param: ExpertParam, env_config: dict[str:int]):
        super().__init__(
            training_param,
            env_config,
            DiscretePointEnvFactory(env_config),
            DiscretePointEnvIdentifierGenerator(),
        )


class ContPointEnvExpertManagerFactory(PointEnvExpertManagerFactory):
    def __init__(self, training_param: ExpertParam, env_config: dict[str:int]):
        super().__init__(
            training_param,
            env_config,
            ContPointEnvFactory(env_config),
            ContPointEnvIdentifierGenerator(),
        )


class PointEnvExpertDefault:
    def __init__(self, expert_managers: list[ExpertManager]):
        self._expert_managers = expert_managers

    def train_and_save_models(self) -> None:
        for expert_manager in self._expert_managers:
            expert_manager.train_and_save_model()

    def save_trajectories(self) -> None:
        for expert_manager in self._expert_managers:
            expert_manager.save_trajectory()

    def save_trajectories_stats(self) -> None:
        for expert_manager in self._expert_managers:
            expert_manager.save_trajectory_stats()

    def show_trajectories_stats(self) -> None:
        TrajectoriesStats(self.load_trajectories()).show_stats()

    def save_trajectories_plot(self):
        for expert_manager in self._expert_managers:
            expert_manager.save_trajectory_plot()

    def show_trajectories_plot_parallel(self, plot_agent_as_hist: bool = False) -> None:
        ParallelTrajectoriesPlot(self.load_trajectories()).show_plot(
            plot_agent_as_hist=plot_agent_as_hist
        )

    def show_trajectories_plot_separate(self, plot_agent_as_hist: bool = False) -> None:
        SeparateTrajectoriesPlot(self.load_trajectories()).show_plot(
            plot_agent_as_hist=plot_agent_as_hist
        )

    def load_trajectories(self) -> list[np.ndarray]:
        trajectories = [
            expert_manager.load_trajectory() for expert_manager in self._expert_managers
        ]

        return trajectories

    def run_models(self):
        model = self._expert_managers[0].load_model()

        class ActionProviderModel(ActionProvider):
            def get_action(self, obs: np.ndarray, **kwargs):
                return model.predict(obs)[0]

        DiscretePointEnvRunner().run_episodes(ActionProviderModel())


class DiscretePointEnvExpertDefault(PointEnvExpertDefault):
    def __init__(self):
        super().__init__(self._make_expert_managers())

    @staticmethod
    def _make_expert_managers() -> list[ExpertManager]:
        training_param = ExpertParam()
        env_configs = PointEnvConfigFactory().env_configs

        return [
            DiscretePointEnvExpertManagerFactory(training_param, env_config).create()
            for env_config in env_configs
        ]

    def run_models(self):
        model = self._expert_managers[0].load_model()

        class ActionProviderModel(ActionProvider):
            def get_action(self, obs: np.ndarray, **kwargs):
                return model.predict(obs)[0]

        DiscretePointEnvRunner().run_episodes(ActionProviderModel())


class ContPointEnvExpertDefault(PointEnvExpertDefault):
    def __init__(self):
        super().__init__(self._make_expert_managers())

    @staticmethod
    def _make_expert_managers() -> list[ExpertManager]:
        training_param = ExpertParam()
        env_configs = PointEnvConfigFactory().env_configs

        return [
            ContPointEnvExpertManagerFactory(training_param, env_config).create()
            for env_config in env_configs
        ]

    def run_models(self):
        model = self._expert_managers[0].load_model()

        class ActionProviderModel(ActionProvider):
            def get_action(self, obs: np.ndarray, **kwargs):
                return model.predict(obs)[0]

        ContPointEnvRunner().run_episodes(ActionProviderModel())


def client_code():
    trainer = ContPointEnvExpertDefault()
    trainer.run_models()


if __name__ == "__main__":
    client_code()

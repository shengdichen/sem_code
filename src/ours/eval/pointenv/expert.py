import numpy as np
from gym import Env

from src.ours.env.creation import (
    DiscretePointNavFactory,
    ContPointNavFactory,
    PointNavFactory,
)
from src.ours.env.config import PointNavConfigFactory
from src.ours.env.identifier import (
    PointNavIdentifierGenerator,
    DiscretePointNavIdentifierGenerator,
    ContPointNavIdentifierGenerator,
)
from src.ours.eval.common.action_provider import ActionProvider
from src.ours.eval.pointenv.run.run import DiscretePointNavRunner, ContPointNavRunner
from src.ours.rl.expert.param import ExpertParam
from src.ours.rl.expert.manager import ExpertManager
from src.ours.rl.expert.sb3.manager import ExpertSb3Manager
from src.ours.rl.common.trajectory.analyzer.plot.multi import (
    ParallelTrajectoriesPlot,
    SeparateTrajectoriesPlot,
)
from src.ours.rl.common.trajectory.analyzer.stats.multi import TrajectoriesStats
from src.ours.rl.expert.trajectory import ExpertTrajectoryManager


class PointNavExpertManagerFactory:
    def __init__(
        self,
        training_param: ExpertParam,
        env_config: dict[str:int],
        env_factory: PointNavFactory,
        env_identifier_generator: PointNavIdentifierGenerator,
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


class DiscretePointNavExpertManagerFactory(PointNavExpertManagerFactory):
    def __init__(self, training_param: ExpertParam, env_config: dict[str:int]):
        super().__init__(
            training_param,
            env_config,
            DiscretePointNavFactory(env_config),
            DiscretePointNavIdentifierGenerator(),
        )


class ContPointNavExpertManagerFactory(PointNavExpertManagerFactory):
    def __init__(self, training_param: ExpertParam, env_config: dict[str:int]):
        super().__init__(
            training_param,
            env_config,
            ContPointNavFactory(env_config),
            ContPointNavIdentifierGenerator(),
        )


class PointNavExpertDefault:
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

        DiscretePointNavRunner().run_episodes(ActionProviderModel())


class DiscretePointNavExpertDefault(PointNavExpertDefault):
    def __init__(self):
        super().__init__(self._make_expert_managers())

    @staticmethod
    def _make_expert_managers() -> list[ExpertManager]:
        training_param = ExpertParam()
        env_configs = PointNavConfigFactory().env_configs

        return [
            DiscretePointNavExpertManagerFactory(training_param, env_config).create()
            for env_config in env_configs
        ]

    def run_models(self):
        model = self._expert_managers[0].load_model()

        class ActionProviderModel(ActionProvider):
            def get_action(self, obs: np.ndarray, **kwargs):
                return model.predict(obs)[0]

        DiscretePointNavRunner().run_episodes(ActionProviderModel())


class ContPointNavExpertDefault(PointNavExpertDefault):
    def __init__(self):
        super().__init__(self._make_expert_managers())

    @staticmethod
    def _make_expert_managers() -> list[ExpertManager]:
        training_param = ExpertParam()
        env_configs = PointNavConfigFactory().env_configs

        return [
            ContPointNavExpertManagerFactory(training_param, env_config).create()
            for env_config in env_configs
        ]

    def run_models(self):
        model = self._expert_managers[0].load_model()

        class ActionProviderModel(ActionProvider):
            def get_action(self, obs: np.ndarray, **kwargs):
                return model.predict(obs)[0]

        ContPointNavRunner().run_episodes(ActionProviderModel())


def client_code():
    trainer = ContPointNavExpertDefault()
    trainer.run_models()


if __name__ == "__main__":
    client_code()

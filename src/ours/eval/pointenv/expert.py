import numpy as np
from gym import Env

from src.ours.env.creation import (
    PointEnvFactory,
    PointEnvIdentifierGenerator,
    PointEnvConfigFactory,
    PointEnvContFactory,
    PointEnvContIdentifierGenerator,
    PointEnvFactoryBase,
    PointEnvIdentifierGeneratorBase,
)
from src.ours.eval.pointenv.run.run import PointEnvRunner
from src.ours.eval.pointenv.run.actionprovider import ActionProvider
from src.ours.util.common.param import ExpertParam
from src.ours.util.expert.manager import ExpertManager
from src.ours.util.expert.sb3.manager import Sb3Manager
from src.ours.util.expert.trajectory.analyzer.plot.multi import (
    TrajectoriesPlotParallel,
    TrajectoriesPlotSeparate,
)
from src.ours.util.expert.trajectory.analyzer.stats.multi import TrajectoriesStats
from src.ours.util.expert.trajectory.manager import TrajectoryManager


class PointEnvExpertManagerFactoryBase:
    def __init__(
        self,
        training_param: ExpertParam,
        env_config: dict[str:int],
        env_factory: PointEnvFactoryBase,
        env_identifier_generator: PointEnvIdentifierGeneratorBase,
    ):
        self._training_param = training_param
        self._env_config = env_config

        self._env_factory = env_factory
        self._env_identifier_generator = env_identifier_generator

    def create(self) -> ExpertManager:
        (env, env_eval), env_identifier = self._get_envs_and_identifier()

        sb3_manager = Sb3Manager(
            ((env, env_eval), env_identifier), self._training_param
        )
        trajectory_manager = TrajectoryManager(
            (env, env_identifier),
            (sb3_manager.model, self._training_param),
        )

        return ExpertManager(
            (sb3_manager, trajectory_manager),
            env_identifier,
        )

    def _get_envs_and_identifier(self) -> tuple[tuple[Env, Env], str]:
        pass


class PointEnvExpertManagerFactory(PointEnvExpertManagerFactoryBase):
    def __init__(self, training_param: ExpertParam, env_config: dict[str:int]):
        super().__init__(
            training_param,
            env_config,
            PointEnvFactory(env_config),
            PointEnvIdentifierGenerator(),
        )

    def _get_envs_and_identifier(self) -> tuple[tuple[Env, Env], str]:
        env, env_eval = (
            PointEnvFactory(self._env_config).create(),
            PointEnvFactory(self._env_config).create(),
        )
        env_identifier = PointEnvIdentifierGenerator().from_env(env)

        return (env, env_eval), env_identifier


class PointEnvContExpertManagerFactory(PointEnvExpertManagerFactoryBase):
    def __init__(self, training_param: ExpertParam, env_config: dict[str:int]):
        super().__init__(
            training_param,
            env_config,
            PointEnvContFactory(env_config),
            PointEnvContIdentifierGenerator(),
        )

    def _get_envs_and_identifier(self) -> tuple[tuple[Env, Env], str]:
        env, env_eval = (
            PointEnvContFactory(self._env_config).create(),
            PointEnvContFactory(self._env_config).create(),
        )
        env_identifier = PointEnvContIdentifierGenerator().from_env(env)

        return (env, env_eval), env_identifier


class PointEnvExpertDefault:
    def __init__(self):
        self._expert_managers = self._make_expert_managers()

    @staticmethod
    def _make_expert_managers() -> list[ExpertManager]:
        training_param = ExpertParam()
        env_configs = PointEnvConfigFactory().env_configs

        return [
            PointEnvExpertManagerFactory(training_param, env_config).create()
            for env_config in env_configs
        ]

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
        TrajectoriesPlotParallel(self.load_trajectories()).show_plot(
            plot_agent_as_hist=plot_agent_as_hist
        )

    def show_trajectories_plot_separate(self, plot_agent_as_hist: bool = False) -> None:
        TrajectoriesPlotSeparate(self.load_trajectories()).show_plot(
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

        PointEnvRunner().run_episodes(ActionProviderModel())


def client_code():
    trainer = PointEnvExpertDefault()
    trainer.run_models()


if __name__ == "__main__":
    client_code()

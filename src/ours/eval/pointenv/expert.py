import numpy as np

from src.ours.env.creation import (
    PointEnvFactory,
    PointEnvIdentifierGenerator,
    PointEnvConfigFactory,
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


class PointEnvExpertManagerFactory:
    def __init__(self, training_param: ExpertParam, env_config: dict[str:int]):
        self._training_param = training_param
        self._env_config = env_config

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

    def train_and_save(self) -> None:
        for expert_manager in self._expert_managers:
            expert_manager.train_model()
            expert_manager.save_model_and_trajectory()

    def show_trajectories_stats(self) -> None:
        TrajectoriesStats(self.load_trajectories()).show_stats()

    def save_trajectories_stats(self) -> None:
        for expert_manager in self._expert_managers:
            expert_manager.save_trajectory_stats()

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

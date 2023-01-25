import numpy as np
from gym import Env
from matplotlib import pyplot as plt
from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.util.common.param import CommonParam
from src.ours.util.expert.path import ExpertSaveLoadPathGenerator
from src.ours.util.common.saveload.path import SaveLoadPathGenerator
from src.ours.util.expert.trajectory.analyzer.plot.single import TrajectoryPlot
from src.ours.util.expert.trajectory.analyzer.stats.single import TrajectoryStats
from src.ours.util.expert.trajectory.util.generator import (
    TrajectoryGeneratorConfig,
    TrajectoryGenerator,
)
from src.ours.util.expert.trajectory.util.saveload import TrajectorySaveLoad


class TrajectoryManager:
    def __init__(
        self,
        env_and_model: tuple[Env, BaseAlgorithm],
        path_generator: SaveLoadPathGenerator,
        trajectory_generator_config: TrajectoryGeneratorConfig,
    ):
        env, model = env_and_model
        self._trajectory_generator = TrajectoryGenerator(
            (env, model), trajectory_generator_config
        )

        path_saveload = path_generator.get_trajectory_path()
        self._trajectory_path = path_saveload / "trajectory.npy"
        self._stats_path = path_saveload / "stats"
        self._plot_path = path_saveload / "plot.png"

    def save(self) -> None:
        trajectory = self._trajectory_generator.get_trajectories()

        TrajectorySaveLoad(self._trajectory_path).save(trajectory)

    def load(self) -> np.ndarray:
        return TrajectorySaveLoad(self._trajectory_path).load()

    def save_stats(self) -> None:
        with open(self._stats_path, "w") as f:
            f.write(TrajectoryStats(self.load()).get_stats())

    def show_stats(self) -> None:
        print(TrajectoryStats(self.load()).get_stats())

    def save_plot(self) -> None:
        figure = plt.figure(figsize=(15, 12), dpi=200)

        TrajectoryPlot(self.load(), figure).plot_agent_target_action()

        figure.savefig(self._plot_path)

        plt.close()

    def show_plot(self) -> None:
        TrajectoryPlot(
            self.load(), plt.figure(figsize=(15, 12), dpi=200)
        ).plot_agent_target_action()

        plt.show()


class ExpertTrajectoryManager(TrajectoryManager):
    def __init__(
        self,
        env_and_identifier: tuple[Env, str],
        model_and_training_param: tuple[BaseAlgorithm, CommonParam],
        trajectory_generator_config=TrajectoryGeneratorConfig(),
    ):
        env, env_identifier = env_and_identifier
        model, training_param = model_and_training_param

        super().__init__(
            (env, model),
            ExpertSaveLoadPathGenerator(env_identifier, training_param),
            trajectory_generator_config,
        )

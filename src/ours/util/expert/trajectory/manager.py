from typing import Any

import numpy as np
from gym import Env
from matplotlib import pyplot as plt

from src.ours.util.common.param import CommonParam
from src.ours.util.common.pathprovider import (
    ExpertSaveLoadPathGenerator,
    SaveLoadPathGeneratorBase,
)
from src.ours.util.expert.trajectory.analyzer.plot.single import TrajectoryPlot
from src.ours.util.expert.trajectory.analyzer.stats.single import TrajectoryStats
from src.ours.util.expert.trajectory.util.generator import (
    TrajectoryGeneratorConfig,
    TrajectoryGenerator,
)
from src.ours.util.expert.trajectory.util.saveload import TrajectorySaveLoad


class TrajectoryManagerBase:
    def __init__(
        self,
        env_and_identifier: tuple[Env, str],
        model_and_training_param: tuple[Any, CommonParam],
        path_generator: SaveLoadPathGeneratorBase,
        trajectory_generator_config=TrajectoryGeneratorConfig(),
    ):
        env, env_identifier = env_and_identifier
        model, training_param = model_and_training_param
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

    def save_plot(self) -> None:
        figure = plt.figure(figsize=(15, 12), dpi=200)

        TrajectoryPlot(self.load(), figure).plot_agent_target_action()

        figure.savefig(self._plot_path)

        plt.close()


class TrajectoryManager(TrajectoryManagerBase):
    def __init__(
        self,
        env_and_identifier: tuple[Env, str],
        model_and_training_param: tuple[Any, CommonParam],
        trajectory_generator_config=TrajectoryGeneratorConfig(),
    ):
        __, env_identifier = env_and_identifier
        __, training_param = model_and_training_param

        super().__init__(
            env_and_identifier,
            model_and_training_param,
            ExpertSaveLoadPathGenerator(env_identifier, training_param),
            trajectory_generator_config,
        )

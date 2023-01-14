from typing import Any

import numpy as np
from gym import Env
from matplotlib import pyplot as plt

from src.ours.util.common.param import CommonParam
from src.ours.util.common.pathprovider import ExpertSaveLoadPathGenerator
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
        env_and_identifier: tuple[Env, str],
        model_and_training_param: tuple[Any, CommonParam],
        trajectory_generator_config=TrajectoryGeneratorConfig(),
    ):
        env, env_identifier = env_and_identifier
        model, training_param = model_and_training_param
        self._trajectory_generator = TrajectoryGenerator(
            (env, model), trajectory_generator_config
        )
        self._path_saveload = ExpertSaveLoadPathGenerator(
            training_param
        ).get_trajectory_path(env_identifier)

    def save(self) -> None:
        trajectory = self._trajectory_generator.get_trajectories()

        TrajectorySaveLoad(self._path_saveload).save(trajectory)

    def load(self) -> np.ndarray:
        return TrajectorySaveLoad(self._path_saveload).load()

    def save_stats(self) -> None:
        with open(self._path_saveload, "w") as f:
            f.write(TrajectoryStats(self.load()).get_stats())

    def save_plot(self) -> None:
        figure = plt.figure(figsize=(15, 12), dpi=200)

        TrajectoryPlot(self.load(), figure).plot_agent_target_action()

        figure.savefig(self._path_saveload)

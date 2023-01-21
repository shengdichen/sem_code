from typing import Any

import numpy as np
from gym import Env
from matplotlib import pyplot as plt

from src.ours.util.common.param import PwilParam
from src.ours.util.common.pathprovider import PwilSaveLoadPathGenerator
from src.ours.util.expert.trajectory.analyzer.plot.single import TrajectoryPlot
from src.ours.util.expert.trajectory.analyzer.stats.single import TrajectoryStats
from src.ours.util.expert.trajectory.manager import TrajectoryManagerBase
from src.ours.util.expert.trajectory.util.generator import (
    TrajectoryGeneratorConfig,
)
from src.ours.util.expert.trajectory.util.saveload import TrajectorySaveLoad


class TrajectoryManager(TrajectoryManagerBase):
    def __init__(
        self,
        env_and_identifier: tuple[Env, str],
        model_and_training_param: tuple[Any, PwilParam],
        trajectory_generator_config=TrajectoryGeneratorConfig(),
    ):
        __, env_identifier = env_and_identifier
        __, training_param = model_and_training_param

        super().__init__(
            env_and_identifier,
            model_and_training_param,
            PwilSaveLoadPathGenerator(env_identifier, training_param),
            trajectory_generator_config,
        )

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

        plt.close()

from typing import Any

import numpy as np
from gym import Env

from src.ours.util.common.param import CommonParam
from src.ours.util.common.pathprovider import TrajectorySaveLoadPathGenerator
from src.ours.util.expert.trajectory.util.saveload import TrajectorySaveLoad
from src.ours.util.expert.trajectory.util.generator import (
    TrajectoryGeneratorConfig,
    TrajectoryGenerator,
)


class TrajectoryManager:
    def __init__(
        self,
        env_and_identifier: tuple[Env, str],
        model_and_training_param: tuple[Any, CommonParam],
        trajectory_generator_config=TrajectoryGeneratorConfig(),
    ):
        env, self._env_identifier = env_and_identifier
        model, training_param = model_and_training_param
        self._trajectory_generator = TrajectoryGenerator(
            (env, model), trajectory_generator_config
        )
        self._path_generator = TrajectorySaveLoadPathGenerator(training_param)

    def save_trajectory(self) -> None:
        trajectory = self._trajectory_generator.get_trajectory()
        path_saveload = self._path_generator.get_path(self._env_identifier)

        TrajectorySaveLoad(path_saveload).save(trajectory)

    def load_trajectory(self) -> np.ndarray:
        path_saveload = self._path_generator.get_path(self._env_identifier)
        return TrajectorySaveLoad(path_saveload).load()

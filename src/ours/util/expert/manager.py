from typing import Any

import numpy as np
from gym import Env

from src.ours.env.creation import PathGenerator
from src.ours.eval.param import CommonParam
from src.ours.util.expert.saveload import ExpertSaveLoad
from src.ours.util.expert.trajectory import (
    TrajectoryGeneratorConfig,
    TrajectoryGenerator,
)
from src.ours.util.pathprovider import ExpertPathGenerator


class ExpertManager:
    def __init__(
        self,
        env_model: tuple[Env, Any],
        training_param: CommonParam,
        expert_manager_param=TrajectoryGeneratorConfig(),
    ):
        self._expert_generator = TrajectoryGenerator(env_model, expert_manager_param)
        self._path_generator = ExpertPathGenerator(training_param)

    def save_expert_traj(self, filename: str) -> None:
        expert_traj = self._expert_generator.get_trajectory()
        path_saveload = self._path_generator.get_path(filename)

        ExpertSaveLoad(path_saveload).save(expert_traj)

    def load_one_demo(self, filename: str) -> np.ndarray:
        path_saveload = self._path_generator.get_path(filename)
        return ExpertSaveLoad(path_saveload).load()

    def load_default_demos(self) -> list[np.ndarray]:
        expert_demos = []
        for env_config in [
            {"n_targets": 2, "shift_x": 0, "shift_y": 0},
            {"n_targets": 2, "shift_x": 0, "shift_y": 50},
            {"n_targets": 2, "shift_x": 50, "shift_y": 0},
        ]:
            filename = PathGenerator(env_config).get_filename_from_shift_values()
            demo = self.load_one_demo(filename)
            expert_demos.append(demo)

        return expert_demos

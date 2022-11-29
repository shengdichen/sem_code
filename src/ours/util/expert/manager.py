from typing import Any

import numpy as np
from gym import Env

from src.ours.env.creation import PointEnvIdentifierGenerator, PointEnvConfigFactory
from src.ours.util.common.param import CommonParam
from src.ours.util.expert.util.saveload import ExpertSaveLoad
from src.ours.util.expert.util.trajectory import (
    TrajectoryGeneratorConfig,
    TrajectoryGenerator,
)
from src.ours.util.common.pathprovider import ExpertSaveLoadPathGenerator


class ExpertManager:
    def __init__(
        self,
        env_model: tuple[Env, Any],
        training_param: CommonParam,
        expert_manager_param=TrajectoryGeneratorConfig(),
    ):
        self._expert_generator = TrajectoryGenerator(env_model, expert_manager_param)
        self._path_generator = ExpertSaveLoadPathGenerator(training_param)

    def save_expert_traj(self, env_identifier: str) -> None:
        expert_traj = self._expert_generator.get_trajectory()
        path_saveload = self._path_generator.get_path(env_identifier)

        ExpertSaveLoad(path_saveload).save(expert_traj)

    def load_one_demo(self, env_identifier: str) -> np.ndarray:
        path_saveload = self._path_generator.get_path(env_identifier)
        return ExpertSaveLoad(path_saveload).load()

    def load_default_demos(self) -> list[np.ndarray]:
        expert_demos = []
        for env_config in PointEnvConfigFactory().env_configs:
            env_identifier = PointEnvIdentifierGenerator().get_identifier(env_config)
            demo = self.load_one_demo(env_identifier)
            expert_demos.append(demo)

        return expert_demos

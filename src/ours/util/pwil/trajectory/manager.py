from gym import Env
from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.util.common.param import PwilParam
from src.ours.util.common.pathprovider import PwilSaveLoadPathGenerator
from src.ours.util.expert.trajectory.manager import TrajectoryManagerBase
from src.ours.util.expert.trajectory.util.generator import (
    TrajectoryGeneratorConfig,
)


class TrajectoryManager(TrajectoryManagerBase):
    def __init__(
        self,
        env_and_identifier: tuple[Env, str],
        model_and_training_param: tuple[BaseAlgorithm, PwilParam],
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

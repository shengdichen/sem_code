from gym import Env
from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.util.common.param import CommonParam
from src.ours.util.expert.path import ExpertSaveLoadPathGenerator
from src.ours.util.expert.trajectory.manager import TrajectoryManager
from src.ours.util.expert.trajectory.util.generator import TrajectoryGeneratorConfig


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

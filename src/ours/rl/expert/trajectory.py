from gym import Env
from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.rl.common.trajectory.manager import TrajectoryManager
from src.ours.rl.common.trajectory.util.generator import TrajectoryGeneratorConfig
from src.ours.rl.expert.param import ExpertParam
from src.ours.rl.expert.path import ExpertSaveLoadPathGenerator


class ExpertTrajectoryManager(TrajectoryManager):
    def __init__(
        self,
        env_and_identifier: tuple[Env, str],
        model_and_training_param: tuple[BaseAlgorithm, ExpertParam],
        trajectory_generator_config=TrajectoryGeneratorConfig(),
    ):
        env, env_identifier = env_and_identifier
        model, training_param = model_and_training_param

        super().__init__(
            (env, model),
            ExpertSaveLoadPathGenerator(env_identifier, training_param),
            trajectory_generator_config,
        )

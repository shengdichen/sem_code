from gym import Env
from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.util.common.param import PwilParam
from src.ours.util.common.pathprovider import PwilSaveLoadPathGenerator
from src.ours.util.common.train import Trainer


class Sb3PwilTrainer(Trainer):
    def __init__(
        self,
        model: BaseAlgorithm,
        training_param: PwilParam,
        env_raw_testing_and_identifier: tuple[Env, str],
    ):
        env_raw_testing, env_identifier = env_raw_testing_and_identifier

        super().__init__(
            model,
            training_param,
            PwilSaveLoadPathGenerator(env_identifier, training_param),
            env_raw_testing,
        )

from gym import Env
from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.util.common.param import CommonParam
from src.ours.util.common.pathprovider import ExpertSaveLoadPathGenerator
from src.ours.util.common.train import Trainer


class Sb3Trainer(Trainer):
    def __init__(
        self,
        model: BaseAlgorithm,
        training_param: CommonParam,
        env_raw_testing_and_identifier: tuple[Env, str] = None,
    ):
        env_raw_testing, env_identifier = env_raw_testing_and_identifier

        super().__init__(
            model,
            training_param,
            ExpertSaveLoadPathGenerator(env_identifier, training_param),
            env_raw_testing,
        )

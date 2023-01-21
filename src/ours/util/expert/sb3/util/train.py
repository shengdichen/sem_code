from gym import Env
from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.util.common.param import CommonParam
from src.ours.util.common.train import Trainer


class Sb3Trainer(Trainer):
    def __init__(
        self,
        model: BaseAlgorithm,
        training_param: CommonParam,
        env_raw_testing_and_identifier: tuple[Env, str] = None,
    ):
        super().__init__(model, training_param, env_raw_testing_and_identifier)

from gym import Env
from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.rl.pwil.param import PwilParam
from src.ours.rl.pwil.path import PwilSaveLoadPathGenerator
from src.ours.rl.common.train import Trainer


class PwilSb3Trainer(Trainer):
    def __init__(
        self,
        model: BaseAlgorithm,
        training_param: PwilParam,
        env_eval_and_identifier: tuple[Env, str],
    ):
        env_eval, env_identifier = env_eval_and_identifier

        super().__init__(
            model,
            training_param,
            PwilSaveLoadPathGenerator(env_identifier, training_param),
            env_eval,
        )

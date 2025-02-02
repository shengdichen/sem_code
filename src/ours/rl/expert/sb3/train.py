from gym import Env
from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.rl.common.train import Trainer
from src.ours.rl.expert.param import ExpertParam
from src.ours.rl.expert.path import ExpertSaveLoadPathGenerator


class ExpertSb3Trainer(Trainer):
    def __init__(
        self,
        model: BaseAlgorithm,
        training_param: ExpertParam,
        env_eval_and_identifier: tuple[Env, str],
    ):
        env_eval, env_identifier = env_eval_and_identifier

        super().__init__(
            model,
            training_param,
            ExpertSaveLoadPathGenerator(env_identifier, training_param),
            env_eval,
        )

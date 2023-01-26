import gym

from src.ours.rl.common.param import CommonParam
from src.ours.rl.common.sb3.manager import Sb3Manager
from src.ours.rl.expert.path import ExpertSaveLoadPathGenerator
from src.ours.rl.expert.sb3.model import ExpertAlgorithmFactory
from src.ours.rl.expert.sb3.train import ExpertSb3Trainer


class ExpertSb3Manager(Sb3Manager):
    def __init__(
        self,
        envs_and_identifier: tuple[tuple[gym.Env, gym.Env], str],
        training_param: CommonParam,
    ):
        (env, __), env_identifier = envs_and_identifier

        super().__init__(
            envs_and_identifier,
            ExpertSaveLoadPathGenerator(env_identifier, training_param),
            ExpertAlgorithmFactory(
                (env, env_identifier), training_param
            ).get_algorithm(),
        )

        self._training_param = training_param

    def _get_trainer(self) -> ExpertSb3Trainer:
        return ExpertSb3Trainer(
            self._model, self._training_param, (self._env_eval, self._env_identifier)
        )

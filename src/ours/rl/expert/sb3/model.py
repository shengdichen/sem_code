from gym import Env

from src.ours.rl.common.param import CommonParam
from src.ours.rl.common.sb3.algorithm import AlgorithmFactory
from src.ours.rl.expert.path import ExpertSaveLoadPathGenerator


class ExpertAlgorithmFactory(AlgorithmFactory):
    def __init__(
        self, env_and_identifier: tuple[Env, str], training_param: CommonParam
    ):
        self._env, self._env_identifier = env_and_identifier
        self._training_param = training_param

        super().__init__(
            self._env,
            training_param,
            ExpertSaveLoadPathGenerator(self._env_identifier, self._training_param),
        )

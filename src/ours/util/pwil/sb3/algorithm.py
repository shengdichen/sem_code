from gym import Env

from src.ours.util.common.sb3.algorithm import AlgorithmFactory
from src.ours.util.pwil.param import PwilParam
from src.ours.util.pwil.path import PwilSaveLoadPathGenerator


class PwilAlgorithFactory(AlgorithmFactory):
    def __init__(self, env_and_identifier: tuple[Env, str], training_param: PwilParam):
        self._env, self._env_identifier = env_and_identifier
        self._training_param = training_param

        super().__init__(
            self._env,
            training_param,
            PwilSaveLoadPathGenerator(self._env_identifier, self._training_param),
        )

from gym import Env

from src.ours.util.pwil.param import PwilParam
from src.ours.util.pwil.path import PwilSaveLoadPathGenerator
from src.ours.util.common.test import PolicyTester
from src.ours.util.expert.sb3.manager import Sb3Manager
from src.ours.util.expert.sb3.util.model import PwilAlgorithFactory
from src.ours.util.pwil.sb3.train import PwilSb3Trainer


class PwilSb3Manager(Sb3Manager):
    def __init__(
        self,
        envs_and_identifier: tuple[tuple[Env, Env], str],
        training_param: PwilParam,
    ):
        (env_pwil_rewarded, __), env_identifier = envs_and_identifier

        super().__init__(
            envs_and_identifier,
            PwilSaveLoadPathGenerator(env_identifier, training_param),
            PwilAlgorithFactory(
                (env_pwil_rewarded, env_identifier), training_param
            ).get_algorithm(),
        )

        self._training_param = training_param

    def _get_trainer(self) -> PwilSb3Trainer:
        return PwilSb3Trainer(
            self._model, self._training_param, (self._env_eval, self._env_identifier)
        )

    def test(self) -> None:
        PolicyTester.test_policy(self._model)

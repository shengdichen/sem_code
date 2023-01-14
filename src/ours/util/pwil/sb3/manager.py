from gym import Env
from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.util.common.param import PwilParam
from src.ours.util.common.pathprovider import PwilSaveLoadPathGenerator
from src.ours.util.common.test import PolicyTester
from src.ours.util.expert.sb3.util.model import AlgorithmFactory
from src.ours.util.expert.sb3.util.saveload import Sb3Saver, Sb3Loader
from src.ours.util.pwil.sb3.train import Sb3PwilTrainer


class Sb3PwilManager:
    def __init__(
        self,
        training_param: PwilParam,
        env_pwil_and_identifier: tuple[tuple[Env, Env], str],
    ):
        (env_pwil_rewarded, env_raw_testing), env_identifier = env_pwil_and_identifier
        self._model = AlgorithmFactory(
            env_pwil_rewarded, training_param
        ).get_algorithm()

        self._trainer = Sb3PwilTrainer(self._model, training_param, env_raw_testing)

        self._path_saveload = PwilSaveLoadPathGenerator(training_param).get_model_path(
            env_identifier
        )

    @property
    def model(self):
        return self._model

    def train(self) -> None:
        self._trainer.train()

    def save(self) -> None:
        saver = Sb3Saver(self._trainer.model, self._path_saveload)
        saver.save_model()

    def load(self, new_env: Env = None) -> BaseAlgorithm:
        return Sb3Loader(self._trainer.model, self._path_saveload).load_model(new_env)

    def test(self):
        model = self.load()
        PolicyTester.test_policy(model)

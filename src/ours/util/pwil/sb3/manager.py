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
        env_pwil_and_identifier: tuple[tuple[Env, Env], str],
        training_param: PwilParam,
    ):
        (env_pwil_rewarded, env_raw_testing), env_identifier = env_pwil_and_identifier
        self._path_saveload = PwilSaveLoadPathGenerator(training_param).get_model_path(
            env_identifier
        )
        self._model = self._get_model(
            AlgorithmFactory(env_pwil_rewarded, training_param).get_algorithm()
        )

        self._trainer = Sb3PwilTrainer(self._model, training_param, env_raw_testing)

    def _get_model(self, algorithm: BaseAlgorithm) -> BaseAlgorithm:
        sb3_loader = Sb3Loader(algorithm, self._path_saveload)
        if sb3_loader.exists():
            return sb3_loader.load()
        else:
            return algorithm

    @property
    def model(self) -> BaseAlgorithm:
        return self._model

    def train(self) -> None:
        self._trainer.train()

    def save(self) -> None:
        saver = Sb3Saver(self._model, self._path_saveload)
        saver.save()

    def test(self) -> None:
        PolicyTester.test_policy(self._model)

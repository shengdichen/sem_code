from gym import Env
from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.util.common.param import PwilParam
from src.ours.util.common.pathprovider import PwilSaveLoadPathGenerator
from src.ours.util.common.test import PolicyTester
from src.ours.util.expert.sb3.util.model import AlgorithPwilFactory
from src.ours.util.expert.sb3.util.saveload import Sb3Saver, Sb3Loader
from src.ours.util.pwil.sb3.train import Sb3PwilTrainer


class Sb3PwilManager:
    def __init__(
        self,
        env_pwil_and_identifier: tuple[tuple[Env, Env], str],
        training_param: PwilParam,
    ):
        (
            self._env,
            self._env_eval,
        ), self._env_identifier = env_pwil_and_identifier
        self._best_sb3_model_path = PwilSaveLoadPathGenerator(
            self._env_identifier, training_param
        ).get_best_sb3_model_path()
        self._latest_sb3_model_path = PwilSaveLoadPathGenerator(
            self._env_identifier, training_param
        ).get_latest_sb3_model_path()
        self._model = self._get_model(
            AlgorithPwilFactory(
                (self._env, self._env_identifier), training_param
            ).get_algorithm()
        )

        self._training_param = training_param

    def _get_model(self, algorithm: BaseAlgorithm) -> BaseAlgorithm:
        sb3_loader = Sb3Loader(algorithm, self._best_sb3_model_path)
        if sb3_loader.exists():
            return sb3_loader.load(self._env)
        else:
            return algorithm

    @property
    def model(self) -> BaseAlgorithm:
        return self._model

    def train(self) -> None:
        trainer = Sb3PwilTrainer(
            self._model, self._training_param, (self._env_eval, self._env_identifier)
        )

        trainer.train()

    def save(self) -> None:
        saver = Sb3Saver(self._model, self._latest_sb3_model_path)
        saver.save()

    def test(self) -> None:
        PolicyTester.test_policy(self._model)

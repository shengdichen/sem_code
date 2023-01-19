import gym
from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.util.common.param import CommonParam
from src.ours.util.common.pathprovider import ExpertSaveLoadPathGenerator
from src.ours.util.expert.sb3.util.model import AlgorithmFactory
from src.ours.util.expert.sb3.util.saveload import Sb3Saver, Sb3Loader
from src.ours.util.expert.sb3.util.train import Sb3Trainer


class Sb3Manager:
    def __init__(
        self, env_and_identifier: tuple[gym.Env, str], training_param: CommonParam
    ):
        self._env, env_identifier = env_and_identifier
        self._path_saveload = ExpertSaveLoadPathGenerator(
            env_identifier, training_param
        ).get_sb3_model_path()
        self._model = self._get_model(
            AlgorithmFactory(
                (self._env, env_identifier), training_param
            ).get_algorithm()
        )

        self._training_param = training_param

    def _get_model(self, algorithm: BaseAlgorithm) -> BaseAlgorithm:
        sb3_loader = Sb3Loader(algorithm, self._path_saveload)
        if sb3_loader.exists():
            return sb3_loader.load(self._env)
        else:
            return algorithm

    @property
    def model(self) -> BaseAlgorithm:
        return self._model

    def train(self) -> None:
        trainer = Sb3Trainer(self._model, self._training_param)
        trainer.train()

    def save(self) -> None:
        saver = Sb3Saver(self._model, self._path_saveload)
        saver.save()

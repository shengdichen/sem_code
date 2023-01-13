import gym
from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.util.common.param import CommonParam
from src.ours.util.common.pathprovider import Sb3SaveLoadPathGenerator
from src.ours.util.expert.sb3.util.model import AlgorithmFactory
from src.ours.util.expert.sb3.util.saveload import Sb3Saver, Sb3Loader
from src.ours.util.expert.sb3.util.train import Sb3Trainer


class Sb3Manager:
    def __init__(
        self, env_and_identifier: tuple[gym.Env, str], training_param: CommonParam
    ):
        env, env_identifier = env_and_identifier
        self._algorithm = AlgorithmFactory(env, training_param).get_algorithm()
        self._model = self._get_model()
        self._trainer = Sb3Trainer(self._model, training_param)
        self._path_saveload = Sb3SaveLoadPathGenerator(training_param).get_path(
            env_identifier
        )

    def _get_model(self):
        sb3_loader = Sb3Loader(self._algorithm, self._path_saveload)
        if sb3_loader.exists():
            return sb3_loader.load_model()
        else:
            return self._algorithm

    @property
    def model(self):
        return self._model

    def train(self) -> None:
        self._trainer.train()

    def save(self) -> None:
        saver = Sb3Saver(self._trainer.model, self._path_saveload)
        saver.save_model()

    def load(self, new_env: gym.Env = None) -> BaseAlgorithm:
        return Sb3Loader(self._trainer.model, self._path_saveload).load_model(new_env)

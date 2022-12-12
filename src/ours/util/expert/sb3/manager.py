import gym
from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.util.common.param import CommonParam
from src.ours.util.common.pathprovider import Sb3SaveLoadPathGenerator
from src.ours.util.expert.sb3.util.saveload import Sb3Saver, Sb3Loader
from src.ours.util.expert.sb3.util.train import Sb3Trainer


class Sb3Manager:
    def __init__(
        self, env_and_identifier: tuple[gym.Env, str], training_param: CommonParam
    ):
        env, self._env_identifier = env_and_identifier
        self._trainer = Sb3Trainer(env, training_param)
        self._path_generator = Sb3SaveLoadPathGenerator(training_param)

    @property
    def model(self):
        return self._trainer.model

    def train(self) -> None:
        self._trainer.train()

    def save(self):
        path_saveload = self._path_generator.get_path(self._env_identifier)
        saver = Sb3Saver(self._trainer.model, path_saveload)
        saver.save_model()

    def load(self, new_env: gym.Env = None) -> BaseAlgorithm:
        path_saveload = self._path_generator.get_path(self._env_identifier)
        return Sb3Loader(self._trainer.model, path_saveload).load_model(new_env)

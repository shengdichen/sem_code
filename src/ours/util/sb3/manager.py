from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.eval.param import CommonParam
from src.ours.util.pathprovider import Sb3SaveLoadPathGenerator
from src.ours.util.sb3.saveload import Sb3Saver


class Sb3Manager:
    def __init__(self, model: BaseAlgorithm, training_param: CommonParam):
        self._model = model
        self._path_generator = Sb3SaveLoadPathGenerator(training_param)

    def save(self, env_identifier: str):
        path_saveload = self._path_generator.get_path(env_identifier)
        saver = Sb3Saver(self._model, path_saveload)
        saver.save_model()

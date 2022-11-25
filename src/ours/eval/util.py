from pathlib import Path

from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.eval.param import CommonParam
from src.ours.util.pathprovider import Sb3SaveLoadPathGenerator


class Saver:
    def __init__(self, model: BaseAlgorithm, savepath_rel: Path):
        self._model = model
        self._savepath_rel = savepath_rel

    def save_model(self):
        self._model.save(self._savepath_rel)


class SaverManager:
    def __init__(self, model: BaseAlgorithm, training_param: CommonParam):
        self._model = model
        self._path_generator = Sb3SaveLoadPathGenerator(training_param)

    def save(self, env_identifier: str):
        path_saveload = self._path_generator.get_path(env_identifier)
        saver = Saver(self._model, path_saveload)
        saver.save_model()

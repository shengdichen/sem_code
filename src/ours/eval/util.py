from pathlib import Path

from stable_baselines3.common.base_class import BaseAlgorithm


class Saver:
    def __init__(self, model: BaseAlgorithm, savepath_rel: Path):
        self._model = model
        self._savepath_rel = savepath_rel

    def save_model(self):
        self._model.save(self._savepath_rel)

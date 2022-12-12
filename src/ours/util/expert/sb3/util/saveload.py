from pathlib import Path

import gym
from stable_baselines3.common.base_class import BaseAlgorithm


class Sb3Saver:
    def __init__(self, model: BaseAlgorithm, savepath_rel: Path):
        self._model = model
        self._savepath_rel = savepath_rel

    def save_model(self):
        self._model.save(self._savepath_rel)


class Sb3Loader:
    def __init__(self, alg: BaseAlgorithm, savepath_rel: Path):
        self._alg = alg
        self._savepath_rel = savepath_rel

    def load_model(self, env: gym.Env = None) -> BaseAlgorithm:
        return self._alg.load(self._savepath_rel, env)

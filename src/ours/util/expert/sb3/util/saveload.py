from pathlib import Path

import gym
from stable_baselines3.common.base_class import BaseAlgorithm


class Sb3Saver:
    def __init__(self, model: BaseAlgorithm, savepath_rel: Path):
        self._model = model
        self._savepath_rel = str(savepath_rel)
        self._make_path_sb3_usable()

    def _make_path_sb3_usable(self) -> None:
        actual_suffix, image_suffix = self._savepath_rel[-4:], ".zip"
        if actual_suffix != image_suffix:
            self._savepath_rel += image_suffix

    def save_model(self, force_resave: bool = False):
        if force_resave:
            self._save()
        elif not Path.exists(Path(self._savepath_rel)):
            self._save()
        else:
            return

    def _save(self):
        self._model.save(self._savepath_rel)


class Sb3Loader:
    def __init__(self, alg: BaseAlgorithm, savepath_rel: Path):
        self._alg = alg
        self._savepath_rel = str(savepath_rel)
        self._make_path_sb3_usable()

    def _make_path_sb3_usable(self) -> None:
        actual_suffix, image_suffix = self._savepath_rel[-4:], ".zip"
        if actual_suffix != image_suffix:
            self._savepath_rel += image_suffix

    def exists(self) -> bool:
        try:
            self.load_model()
        except FileNotFoundError:
            return False
        else:
            return True

    def load_model(self, new_env: gym.Env = None) -> BaseAlgorithm:
        return self._alg.load(self._savepath_rel, new_env)

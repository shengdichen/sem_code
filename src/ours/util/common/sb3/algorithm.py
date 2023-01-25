from gym import Env
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.util.common.param import CommonParam
from src.ours.util.common.saveload.path import SaveLoadPathGenerator


class AlgorithmFactory:
    def __init__(
        self,
        env: Env,
        training_param: CommonParam,
        saveload_path_generator: SaveLoadPathGenerator,
    ):
        self._env = env
        self._training_param = training_param
        self._tensorboard_log_dir = str(
            saveload_path_generator.get_model_log_path(False)
        )

    def get_algorithm(self) -> BaseAlgorithm:
        return PPO(
            "MlpPolicy",
            self._env,
            verbose=0,
            **self._training_param.kwargs_ppo,
            tensorboard_log=self._tensorboard_log_dir
        )

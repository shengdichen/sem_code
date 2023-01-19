from gym import Env
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.util.common.param import CommonParam, PwilParam
from src.ours.util.common.pathprovider import PwilSaveLoadPathGenerator


class AlgorithmFactory:
    def __init__(
        self, env_and_identifier: tuple[Env, str], training_param: CommonParam
    ):
        self._env, self._env_identifier = env_and_identifier
        self._training_param = training_param

    def get_algorithm(self) -> BaseAlgorithm:
        return PPO(
            "MlpPolicy",
            self._env,
            verbose=0,
            **self._training_param.kwargs_ppo,
            tensorboard_log=self._training_param.sb3_tblog_dir
        )


class AlgorithPwilFactory:
    def __init__(self, env_and_identifier: tuple[Env, str], training_param: PwilParam):
        self._env, self._env_identifier = env_and_identifier
        self._training_param = training_param

    def get_algorithm(self) -> BaseAlgorithm:
        model_path = PwilSaveLoadPathGenerator(
            self._env_identifier, self._training_param
        ).get_model_path()
        model_dir = str(model_path) + "/log/"

        return PPO(
            "MlpPolicy",
            self._env,
            verbose=0,
            **self._training_param.kwargs_ppo,
            tensorboard_log=model_dir
        )

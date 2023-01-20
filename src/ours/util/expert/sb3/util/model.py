from gym import Env
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.util.common.param import CommonParam, PwilParam
from src.ours.util.common.pathprovider import (
    PwilSaveLoadPathGenerator,
    ExpertSaveLoadPathGenerator,
)


class AlgorithmFactoryBase:
    def __init__(self, env: Env, training_param: CommonParam, model_path: str):
        self._env = env
        self._training_param = training_param
        self._model_dir = model_path + "/log/"

    def get_algorithm(self) -> BaseAlgorithm:
        return PPO(
            "MlpPolicy",
            self._env,
            verbose=0,
            **self._training_param.kwargs_ppo,
            tensorboard_log=self._model_dir
        )


class AlgorithmFactory(AlgorithmFactoryBase):
    def __init__(
        self, env_and_identifier: tuple[Env, str], training_param: CommonParam
    ):
        super().__init__()

        self._env, self._env_identifier = env_and_identifier
        self._training_param = training_param

    def get_algorithm(self) -> BaseAlgorithm:
        model_path = ExpertSaveLoadPathGenerator(
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


class AlgorithPwilFactory(AlgorithmFactoryBase):
    def __init__(self, env_and_identifier: tuple[Env, str], training_param: PwilParam):
        super().__init__()

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

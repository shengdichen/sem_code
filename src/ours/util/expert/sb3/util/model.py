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
        self._env, self._env_identifier = env_and_identifier
        self._training_param = training_param

        super().__init__(
            self._env,
            training_param,
            str(
                ExpertSaveLoadPathGenerator(
                    self._env_identifier, self._training_param
                ).get_model_path()
            ),
        )


class AlgorithPwilFactory(AlgorithmFactoryBase):
    def __init__(self, env_and_identifier: tuple[Env, str], training_param: PwilParam):
        self._env, self._env_identifier = env_and_identifier
        self._training_param = training_param

        super().__init__(
            self._env,
            training_param,
            str(
                PwilSaveLoadPathGenerator(
                    self._env_identifier, self._training_param
                ).get_model_path()
            ),
        )

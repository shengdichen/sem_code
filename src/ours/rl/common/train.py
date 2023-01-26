from gym import Env
from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.rl.common.helper import TqdmCallback
from src.ours.rl.common.param import CommonParam
from src.ours.rl.common.saveload.path import SaveLoadPathGenerator
from src.ours.rl.pwil.sb3.util import CallbackListFactory


class Trainer:
    def __init__(
        self,
        model: BaseAlgorithm,
        training_param: CommonParam,
        path_generator: SaveLoadPathGenerator,
        env_eval: Env,
    ):
        self._model = model
        self._training_param = training_param
        if env_eval is not None:
            self._callback_list = CallbackListFactory(
                env_eval,
                path_generator,
            ).callback_list
        else:
            self._callback_list = [TqdmCallback()]

    @property
    def model(self):
        return self._model

    def train(self) -> None:
        self._model.learn(
            total_timesteps=self._training_param.n_steps_training,
            callback=self._callback_list,
        )

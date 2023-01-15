from gym import Env
from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.util.common.param import PwilParam
from src.ours.util.common.train import Trainer
from src.ours.util.pwil.sb3.util import CallbackListFactory


class Sb3PwilTrainer(Trainer):
    def __init__(
        self,
        model: BaseAlgorithm,
        training_param: PwilParam,
        env_raw_testing: Env,
    ):
        self._model = model
        self._training_param = training_param

        self._callback_list = CallbackListFactory(
            training_param, env_raw_testing
        ).callback_list

    @property
    def model(self):
        return self._model

    def train(self) -> None:
        self._model.learn(
            total_timesteps=self._training_param.n_steps_pwil_train,
            callback=self._callback_list,
        )

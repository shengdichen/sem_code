from gym import Env

from src.ours.util.common.param import PwilParam
from src.ours.util.common.train import Trainer
from src.ours.util.expert.sb3.util.model import AlgorithmFactory
from src.ours.util.pwil.sb3.util import CallbackListFactory


class Sb3PwilTrainer(Trainer):
    def __init__(
        self,
        training_param: PwilParam,
        env_pwil_and_testing: tuple[Env, Env],
    ):
        self._training_param = training_param
        env_pwil_rewarded, env_raw_testing = env_pwil_and_testing

        self._model = AlgorithmFactory(
            env_pwil_rewarded, training_param
        ).get_algorithm()

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

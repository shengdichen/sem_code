from gym import Env
from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.util.common.param import PwilParam
from src.ours.util.common.pathprovider import PwilSaveLoadPathGenerator
from src.ours.util.common.train import Trainer
from src.ours.util.pwil.sb3.util import CallbackListFactory


class Sb3PwilTrainer(Trainer):
    def __init__(
        self,
        model: BaseAlgorithm,
        training_param: PwilParam,
        env_raw_testing_and_identifier: tuple[Env, str],
    ):
        super().__init__(model, training_param, env_raw_testing_and_identifier)

        self._model = model
        self._training_param = training_param
        env_raw_testing, env_identifier = env_raw_testing_and_identifier

        self._callback_list = CallbackListFactory(
            env_raw_testing,
            PwilSaveLoadPathGenerator(env_identifier, self._training_param),
        ).callback_list

    @property
    def model(self):
        return self._model

    def train(self) -> None:
        self._model.learn(
            total_timesteps=self._training_param.n_steps_training,
            callback=self._callback_list,
        )

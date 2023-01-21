from gym import Env
from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.util.common.helper import TqdmCallback
from src.ours.util.common.param import CommonParam
from src.ours.util.common.pathprovider import ExpertSaveLoadPathGenerator
from src.ours.util.common.train import Trainer
from src.ours.util.pwil.sb3.util import CallbackListFactory


class Sb3Trainer(Trainer):
    def __init__(
        self,
        model: BaseAlgorithm,
        training_param: CommonParam,
        env_raw_testing_and_identifier: tuple[Env, str] = None,
    ):
        self._model = model
        self._training_param = training_param
        env_raw_testing, env_identifier = env_raw_testing_and_identifier

        if env_raw_testing_and_identifier is not None:
            self._callback_list = CallbackListFactory(
                env_raw_testing,
                ExpertSaveLoadPathGenerator(env_identifier, self._training_param),
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

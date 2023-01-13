from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.util.common.helper import TqdmCallback
from src.ours.util.common.param import CommonParam
from src.ours.util.common.train import Trainer


class Sb3Trainer(Trainer):
    def __init__(self, model: BaseAlgorithm, training_param: CommonParam):
        self._model = model
        self._training_param = training_param

    @property
    def model(self):
        return self._model

    def train(self) -> None:
        self._model.learn(
            total_timesteps=self._training_param.n_steps_expert_train,
            callback=[TqdmCallback()],
        )

from src.ours.eval.param import CommonParam


class Trainer:
    def __init__(self, training_param: CommonParam):
        self._training_param = training_param
        self._sb3_tblog_dir = self._training_param.sb3_tblog_dir
        self._kwargs_ppo = self._training_param.kwargs_ppo

    def train(self, **kwargs):
        pass

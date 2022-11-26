from src.ours.util.common.param import CommonParam


class Trainer:
    def __init__(self, training_param: CommonParam):
        self._training_param = training_param

    def train(self, **kwargs):
        pass

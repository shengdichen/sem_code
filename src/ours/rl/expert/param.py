from src.ours.rl.common.param import CommonParam, Util


class ExpertParam(CommonParam):
    def __init__(self):
        super().__init__(int(1e6))

        self._model_dir, self._demo_dir = "./expert/models/", "./expert/demos/"
        Util.mkdir_if_not_existent([self._model_dir, self._demo_dir])

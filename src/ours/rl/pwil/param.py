from src.ours.rl.common.param import CommonParam, Util


class PwilParam(CommonParam):
    def __init__(self):
        super().__init__(int(5e5))

        self._model_dir, self._demo_dir, self._rewardplot_dir = (
            "./pwil/models/",
            "./pwil/demos/",
            "./pwil/rewardplot/",
        )
        Util.mkdir_if_not_existent(
            [self._model_dir, self._demo_dir, self._rewardplot_dir]
        )

        self._pwil_training_param = {
            "n_demos": 5,
            "subsampling": 10,
            "use_actions": True,
        }

        self._trajectory_num = 0

    @property
    def rewardplot_dir(self):
        return self._rewardplot_dir

    @property
    def pwil_training_param(self):
        return self._pwil_training_param

    @pwil_training_param.setter
    def pwil_training_param(self, value):
        self._pwil_training_param = value

    @property
    def trajectory_num(self):
        return self._trajectory_num

    @trajectory_num.setter
    def trajectory_num(self, value: int):
        self._trajectory_num = value

    def print_pwil_related_info(self):
        print(
            "(demo_id, n_demos, subsampling) := ({0}, {1}, {2})".format(
                self._trajectory_num,
                self._pwil_training_param["n_demos"],
                self._pwil_training_param["subsampling"],
            )
        )

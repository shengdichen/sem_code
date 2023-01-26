from pathlib import Path

from src.ours.rl.common.param import Util
from src.ours.rl.common.saveload.path import SaveLoadPathGenerator
from src.ours.rl.pwil.param import PwilParam


class PwilSaveLoadPathGenerator(SaveLoadPathGenerator):
    def __init__(self, env_identifier: str, training_param: PwilParam):
        super().__init__()

        self._env_identifier = env_identifier
        self._training_param = training_param

        self._trajectory_num_category = self._get_trajectory_num_category()

    def _get_trajectory_num_category(self) -> str:
        if self._training_param.trajectory_num == 0:
            return "optimal"
        elif self._training_param.trajectory_num <= 3:
            return "mixed"
        else:
            return "distant"

    def _get_model_path(self) -> Path:
        return Path(self._get_model_dependent_path(self._training_param.model_dir))

    def get_trajectory_path(self) -> Path:
        path = self._get_model_dependent_path(self._training_param.demo_dir)
        Util.mkdir_if_not_existent([path])

        return Path(path)

    def _get_model_dependent_path(self, raw_dir: str) -> str:
        n_demos = self._training_param.pwil_training_param["n_demos"]
        subsampling = self._training_param.pwil_training_param["subsampling"]

        filename = "{0}_{1:02}_{2:03}".format(
            self._training_param.trajectory_num,
            n_demos,
            subsampling,
        )

        return "{0}/{1}/{2}".format(
            self._get_curr_model_dir(raw_dir), self._trajectory_num_category, filename
        )

    def _get_curr_model_dir(self, raw_dir: str) -> str:
        curr_model_dir = "{0}/{1}{2}{3:07}/".format(
            raw_dir,
            self._env_identifier,
            "_",
            self._training_param.n_steps_training,
        )
        Util.mkdir_if_not_existent([curr_model_dir])

        return curr_model_dir

    def get_rewardplot_path(self) -> Path:
        return self._get_model_independent_path(self._training_param.rewardplot_dir)

    def _get_model_independent_path(self, raw_dir: str) -> Path:
        curr_dir = "{0}/{1}/{2}".format(
            raw_dir, self._env_identifier, self._trajectory_num_category
        )
        Util.mkdir_if_not_existent([curr_dir])

        n_demos = self._training_param.pwil_training_param["n_demos"]
        subsampling = self._training_param.pwil_training_param["subsampling"]

        filename = "{0}_{1:02}_{2:03}".format(
            self._training_param.trajectory_num,
            n_demos,
            subsampling,
        )

        return Path("{0}/{1}".format(curr_dir, filename))

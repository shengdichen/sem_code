from pathlib import Path

from src.ours.util.common.param import CommonParam, PwilParam, Util


class TrajectorySaveLoadPathGenerator:
    def __init__(self, training_param: CommonParam):
        self._training_param = training_param
        self._np_postfix = ".npy"

    def get_path(self, env_identifier: str) -> Path:
        return Path(
            "{0}/{1}{2}{3:07}{4}".format(
                self._training_param.demo_dir,
                env_identifier,
                "_",
                self._training_param.n_steps_expert_train,
                self._np_postfix,
            )
        )


class Sb3SaveLoadPathGenerator:
    def __init__(self, training_param: CommonParam):
        self._training_param = training_param

    def get_path(self, env_identifier: str) -> Path:
        return Path(
            "{0}/{1}{2}{3:07}".format(
                self._training_param.model_dir,
                env_identifier,
                "_",
                self._training_param.n_steps_expert_train,
            )
        )


class PwilSaveLoadPathGenerator:
    def __init__(self, training_param: PwilParam):
        self._training_param = training_param

    def get_model_path(self, env_identifier: str) -> Path:
        return self._get_model_dependent_path(
            self._training_param.model_dir, env_identifier
        )

    def get_trajectory_path(self, env_identifier: str) -> Path:
        return self._get_model_dependent_path(
            self._training_param.demo_dir, env_identifier
        )

    def get_rewardplot_path(self, env_identifier: str) -> Path:
        return self._get_model_dependent_path(
            self._training_param.rewardplot_dir, env_identifier
        )

    def _get_model_dependent_path(self, raw_dir: str, env_identifier: str) -> Path:
        n_demos = self._training_param.pwil_training_param["n_demos"]
        subsampling = self._training_param.pwil_training_param["subsampling"]

        return Path(
            "{0}/{1}_{2:02}_{3:03}".format(
                self._get_curr_model_dir(raw_dir, env_identifier),
                self._training_param.trajectory_num,
                n_demos,
                subsampling,
            )
        )

    def _get_curr_model_dir(self, raw_dir: str, env_identifier: str) -> str:
        curr_model_dir = "{0}/{1}{2}{3:07}/".format(
            raw_dir,
            env_identifier,
            "_",
            self._training_param.n_steps_pwil_train,
        )
        Util.mkdir_if_not_existent([curr_model_dir])

        return curr_model_dir

    def _get_model_independent_path(self, raw_dir: str) -> Path:
        n_demos = self._training_param.pwil_training_param["n_demos"]
        subsampling = self._training_param.pwil_training_param["subsampling"]

        return Path(
            "{0}/{1}_{2:02}_{3:03}".format(
                raw_dir,
                self._training_param.trajectory_num,
                n_demos,
                subsampling,
            )
        )

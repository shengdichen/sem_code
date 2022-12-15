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

    def get_path(self, env_identifier: str, trajectory_num: int = 0) -> Path:
        n_demos = self._training_param.pwil_training_param["n_demos"]
        subsampling = self._training_param.pwil_training_param["subsampling"]

        curr_model_dir = "{0}/{1}{2}{3:07}/".format(
            self._training_param.model_dir,
            env_identifier,
            "_",
            self._training_param.n_steps_expert_train,
        )
        Util.mkdir_if_not_existent([curr_model_dir])

        return Path(
            "{0}/{1}_{2:02}_{3:03}".format(
                curr_model_dir,
                trajectory_num,
                n_demos,
                subsampling,
            )
        )

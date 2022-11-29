from pathlib import Path

from src.ours.util.common.param import CommonParam


class ExpertSaveLoadPathGenerator:
    def __init__(self, training_param: CommonParam):
        self._training_param = training_param

    def get_path(self, env_identifier: str) -> Path:
        return Path(
            "{0}/{1}{2}{3:07}{4}".format(
                self._training_param.demo_dir,
                env_identifier,
                "_",
                self._training_param.n_steps_expert_train,
                ".npy"
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

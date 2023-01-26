from pathlib import Path

from src.ours.rl.common.param import CommonParam, Util
from src.ours.rl.common.saveload.path import SaveLoadPathGenerator


class ExpertSaveLoadPathGenerator(SaveLoadPathGenerator):
    def __init__(self, env_identifier: str, training_param: CommonParam):
        super().__init__()

        self._env_identifier = env_identifier
        self._training_param = training_param

    def _get_model_path(self) -> Path:
        return Path(self._get_model_dependent_path(self._training_param.model_dir))

    def get_trajectory_path(self) -> Path:
        path = self._get_model_dependent_path(self._training_param.demo_dir)
        Util.mkdir_if_not_existent([path])

        return Path(path)

    def _get_model_dependent_path(self, raw_dir: str) -> str:
        filename = "{0}{1}{2:07}".format(
            self._env_identifier,
            "_",
            self._training_param.n_steps_training,
        )

        return "{0}/{1}".format(raw_dir, filename)

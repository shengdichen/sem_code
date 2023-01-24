from pathlib import Path

from src.ours.util.common.param import CommonParam, PwilParam, Util


class SaveLoadPathGeneratorBase:
    def get_best_sb3_model_path(self) -> Path:
        return self.get_model_eval_path() / "best_model.zip"

    def get_model_eval_path(self) -> Path:
        return self._get_model_path() / "eval"

    def get_latest_sb3_model_path(self) -> Path:
        return self._get_model_path() / "latest.zip"

    def get_model_log_path(self, use_simple_log: bool) -> Path:
        log_path = self._get_model_path() / "log"

        if use_simple_log:
            return log_path / "simple"
        else:
            return log_path

    def _get_model_path(self) -> Path:
        pass

    def get_trajectory_path(self) -> Path:
        pass

    def _get_model_dependent_path(self, raw_dir: str) -> str:
        pass


class ExpertSaveLoadPathGenerator(SaveLoadPathGeneratorBase):
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


class PwilSaveLoadPathGenerator(SaveLoadPathGeneratorBase):
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
        n_demos = self._training_param.pwil_training_param["n_demos"]
        subsampling = self._training_param.pwil_training_param["subsampling"]

        filename = "{0}_{1:02}_{2:03}".format(
            self._training_param.trajectory_num,
            n_demos,
            subsampling,
        )

        return Path(
            "{0}/{1}/{2}/{3}".format(
                raw_dir, self._env_identifier, self._trajectory_num_category, filename
            )
        )

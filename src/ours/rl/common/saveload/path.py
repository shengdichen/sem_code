from pathlib import Path


class SaveLoadPathGenerator:
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

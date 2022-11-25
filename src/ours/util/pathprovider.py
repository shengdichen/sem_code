from pathlib import Path

from src.ours.eval.param import CommonParam


class ExpertPathGenerator:
    def __init__(self, training_param: CommonParam):
        self._training_param = training_param

    def get_path(self, filename: str) -> Path:
        return Path(
            "{0}/{1}{2}{3}".format(
                self._training_param.demo_dir,
                filename,
                self._training_param.n_steps_expert_train,
                self._training_param.postfix,
            )
        )

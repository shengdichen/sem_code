from gym import Env
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

from src.ours.util.common.helper import TqdmCallback
from src.ours.util.common.pathprovider import (
    SaveLoadPathGeneratorBase,
)
from src.upstream.utils import CustomCallback


class CallbackListFactory:
    def __init__(
        self,
        env_raw_testing: Env,
        saveload_path_generator: SaveLoadPathGeneratorBase,
    ):
        self._env_raw_testing = env_raw_testing

        self._log_path = saveload_path_generator.get_model_log_path(True)
        self._eval_path = str(saveload_path_generator.get_model_eval_path())

        self._callback_list = self._make_callback_list()

    @property
    def callback_list(self):
        return self._callback_list

    def _make_callback_list(self) -> CallbackList:
        callback_list = CallbackList(
            [
                CustomCallback(id="", log_path=self._log_path),
                self._make_eval_callback(),
                TqdmCallback(),
            ]
        )

        return callback_list

    def _make_eval_callback(self) -> EvalCallback:
        eval_callback = EvalCallback(
            self._env_raw_testing,
            best_model_save_path=self._eval_path,
            log_path=self._eval_path,
            eval_freq=10000,
            deterministic=False,
            render=False,
        )

        # eval_callback.init_callback(ppo_dict[k])
        return eval_callback

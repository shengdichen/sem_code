from gym import Env
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

from src.ours.util.common.helper import TqdmCallback
from src.ours.util.common.param import PwilParam
from src.ours.util.common.pathprovider import PwilSaveLoadPathGenerator
from src.upstream.utils import CustomCallback


class CallbackListFactory:
    def __init__(
        self,
        training_param: PwilParam,
        env_raw_testing_and_identifier: tuple[Env, str],
    ):
        self._training_param = training_param
        self._env_raw_testing, self._env_identifier = env_raw_testing_and_identifier

        self._callback_list = self._make_callback_list()

    @property
    def callback_list(self):
        return self._callback_list

    def _make_callback_list(self) -> CallbackList:
        model_path = PwilSaveLoadPathGenerator(
            self._env_identifier, self._training_param
        ).get_model_path()
        log_path = str(model_path) + "/log/simple/"

        callback_list = CallbackList(
            [
                CustomCallback(id="", log_path=log_path),
                self._make_eval_callback(),
                TqdmCallback(),
            ]
        )

        return callback_list

    def _make_eval_callback(self) -> EvalCallback:
        model_path = PwilSaveLoadPathGenerator(
            self._env_identifier, self._training_param
        ).get_model_path()
        eval_path = str(model_path) + "/eval/"

        eval_callback = EvalCallback(
            self._env_raw_testing,
            best_model_save_path=eval_path,
            log_path=eval_path,
            eval_freq=10000,
            deterministic=True,
            render=False,
        )

        # eval_callback.init_callback(ppo_dict[k])
        return eval_callback

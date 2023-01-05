from gym import Env
from stable_baselines3 import PPO as PPOSB
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

from src.ours.util.common.helper import TqdmCallback
from src.ours.util.common.param import PwilParam
from src.upstream.utils import CustomCallback


class CallbackListFactory:
    def __init__(
        self,
        training_param: PwilParam,
        env_raw_testing: Env,
    ):
        self._training_param = training_param
        self._env_raw_testing = env_raw_testing

        self._callback_list = self._make_callback_list()

    @property
    def callback_list(self):
        return self._callback_list

    def _make_callback_list(self) -> CallbackList:
        callback_list = CallbackList(
            [
                CustomCallback(id="", log_path=self._training_param.sb3_tblog_dir),
                self._make_eval_callback(),
                TqdmCallback(),
            ]
        )

        return callback_list

    def _make_eval_callback(self) -> EvalCallback:
        eval_callback = EvalCallback(
            self._env_raw_testing,
            best_model_save_path=self._training_param.sb3_tblog_dir,
            log_path=self._training_param.sb3_tblog_dir,
            eval_freq=10000,
            deterministic=True,
            render=False,
        )

        # eval_callback.init_callback(ppo_dict[k])
        return eval_callback


class PwilModelFactory:
    def __init__(
        self,
        training_param: PwilParam,
        env_pwil_rewarded: Env,
    ):
        self._training_param = training_param
        self._env_pwil_rewarded = env_pwil_rewarded

        self._model = self._make_model()

    @property
    def model(self):
        return self._model

    def _make_model(self) -> BaseAlgorithm:
        model = PPOSB(
            "MlpPolicy",
            self._env_pwil_rewarded,
            verbose=0,
            **self._training_param.kwargs_ppo,
            tensorboard_log=self._training_param.sb3_tblog_dir,
        )

        return model

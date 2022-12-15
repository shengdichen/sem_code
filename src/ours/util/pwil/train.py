import os

import numpy as np
from gym import Env
from stable_baselines3 import PPO as PPOSB
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from src.ours.util.common.helper import RewardPlotter, TqdmCallback
from src.ours.util.common.param import PwilParam
from src.ours.util.common.train import Trainer
from src.ours.util.expert.trajectory.manager import TrajectoryManager
from src.upstream.env_utils import PWILReward
from src.upstream.utils import CustomCallback


class TrainerPwil(Trainer):
    def __init__(
        self,
        training_param: PwilParam,
        envs_and_identifier: tuple[tuple[Env, Env], str],
        trajectories: list[np.ndarray],
    ):
        self._training_param = training_param

        self._model_dir = self._training_param.model_dir
        self._save_deterministic = False

        (
            self._env_raw,
            self._env_raw_testing,
        ), self._env_identifier = envs_and_identifier
        self._trajectories = trajectories

        self._env = self._make_env()

        self._callback_list = self._make_callback_list()

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

    def _make_env(self) -> Env:
        env = PWILReward(
            env=self._env_raw,
            demos=self._trajectories,
            **self._training_param.pwil_training_param,
        )

        return env

    def train(self, fname):
        plot = RewardPlotter.plot_reward(discriminator=None, env=self._env)

        model = PPOSB(
            "MlpPolicy",
            self._env,
            verbose=0,
            **self._training_param.kwargs_ppo,
            tensorboard_log=self._training_param.sb3_tblog_dir,
        )

        model.learn(
            total_timesteps=self._training_param.n_steps_expert_train,
            callback=self._callback_list,
        )

        model.save(
            os.path.join(
                self._model_dir,
                "model_" + fname + str(int(self._training_param.n_steps_expert_train)),
            )
        )
        TrajectoryManager(
            (self._env, self._env_identifier), (model, self._training_param)
        ).save_trajectory()

        return model, plot

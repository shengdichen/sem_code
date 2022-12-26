import numpy as np
from gym import Env
from stable_baselines3 import PPO as PPOSB
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from src.ours.util.common.helper import RewardPlotter, TqdmCallback
from src.ours.util.common.param import PwilParam
from src.ours.util.common.pathprovider import PwilSaveLoadPathGenerator
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

        self._save_deterministic = False

        (
            env_raw,
            env_raw_testing,
        ), self._env_identifier = envs_and_identifier

        self._env = self._make_env(env_raw, trajectories)
        self._model = self._make_model()

        self._callback_list = self._make_callback_list(env_raw_testing)

    @property
    def model(self):
        return self._model

    def _make_callback_list(self, env_raw_testing) -> CallbackList:
        callback_list = CallbackList(
            [
                CustomCallback(id="", log_path=self._training_param.sb3_tblog_dir),
                self._make_eval_callback(env_raw_testing),
                TqdmCallback(),
            ]
        )

        return callback_list

    def _make_eval_callback(self, env_raw_testing) -> EvalCallback:
        eval_callback = EvalCallback(
            env_raw_testing,
            best_model_save_path=self._training_param.sb3_tblog_dir,
            log_path=self._training_param.sb3_tblog_dir,
            eval_freq=10000,
            deterministic=True,
            render=False,
        )

        # eval_callback.init_callback(ppo_dict[k])
        return eval_callback

    def _make_env(self, env_raw: Env, trajectories: list[np.ndarray]) -> Env:
        env = PWILReward(
            env=env_raw,
            demos=trajectories,
            **self._training_param.pwil_training_param,
        )

        return env

    def _make_model(self) -> BaseAlgorithm:
        model = PPOSB(
            "MlpPolicy",
            self._env,
            verbose=0,
            **self._training_param.kwargs_ppo,
            tensorboard_log=self._training_param.sb3_tblog_dir,
        )

        return model

    def get_reward_plot(self) -> np.ndarray:
        plot = RewardPlotter.plot_reward(discriminator=None, env=self._env)

        return plot

    def train(self) -> None:
        self._model.learn(
            total_timesteps=self._training_param.n_steps_expert_train,
            callback=self._callback_list,
        )

    def save_model(self) -> None:
        path = PwilSaveLoadPathGenerator(self._training_param).get_path(
            self._env_identifier
        )
        self._model.save(path)

    def save_trajectory(self) -> None:
        TrajectoryManager(
            (self._env, self._env_identifier), (self._model, self._training_param)
        ).save_trajectory()

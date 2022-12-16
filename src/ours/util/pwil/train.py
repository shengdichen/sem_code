import numpy as np
from gym import Env
from stable_baselines3 import PPO as PPOSB
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from src.ours.util.common.helper import RewardPlotter, TqdmCallback
from src.ours.util.common.param import PwilParam
from src.ours.util.common.pathprovider import PwilSaveLoadPathGenerator
from src.ours.util.common.test import PolicyTester
from src.ours.util.common.train import Trainer
from src.ours.util.expert.sb3.util.saveload import Sb3Saver, Sb3Loader
from src.ours.util.expert.trajectory.manager import TrajectoryManager
from src.upstream.env_utils import PWILReward
from src.upstream.utils import CustomCallback


class PwilEnvFactory:
    def __init__(
        self,
        training_param: PwilParam,
        env_raw: Env,
        trajectories: list[np.ndarray],
    ):
        self._training_param = training_param
        self._env_raw = env_raw

        self._env_pwil_rewarded = self._make_env_pwil_rewarded(trajectories)

    @property
    def env_pwil_rewarded(self) -> Env:
        return self._env_pwil_rewarded

    def _make_env_pwil_rewarded(self, trajectories: list[np.ndarray]) -> Env:
        env_pwil_rewarded = PWILReward(
            env=self._env_raw,
            demos=trajectories,
            **self._training_param.pwil_training_param,
        )

        return env_pwil_rewarded


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


class RewardPlotManager:
    def __init__(self, env_pwil_rewarded: Env):
        self._env_pwil_rewarded = env_pwil_rewarded

    def get_reward_plot(self) -> np.ndarray:
        plot = RewardPlotter.plot_reward(
            discriminator=None, env=self._env_pwil_rewarded
        )

        return plot


class Sb3PwilTrainer(Trainer):
    def __init__(
        self,
        training_param: PwilParam,
        envs_and_identifier: tuple[tuple[Env, Env], str],
        trajectories: list[np.ndarray],
    ):
        self._training_param = training_param

        (
            env_raw,
            env_raw_testing,
        ), self._env_identifier = envs_and_identifier

        self._env_pwil_rewarded = PwilEnvFactory(
            training_param, env_raw, trajectories
        ).env_pwil_rewarded
        self._model = PwilModelFactory(training_param, self._env_pwil_rewarded).model

        self._callback_list = CallbackListFactory(
            training_param, env_raw_testing
        ).callback_list

    @property
    def model(self):
        return self._model

    def train(self) -> None:
        self._model.learn(
            total_timesteps=self._training_param.n_steps_expert_train,
            callback=self._callback_list,
        )

    def save_trajectory(self) -> None:
        TrajectoryManager(
            (self._env_pwil_rewarded, self._env_identifier),
            (self._model, self._training_param),
        ).save_trajectory()


class Sb3PwilManager:
    def __init__(
        self,
        training_param: PwilParam,
        envs_and_identifier: tuple[tuple[Env, Env], str],
        trajectories: list[np.ndarray],
    ):
        self._trainer = Sb3PwilTrainer(
            training_param, envs_and_identifier, trajectories
        )

        env_identifier = envs_and_identifier[1]
        self._path_saveload = PwilSaveLoadPathGenerator(training_param).get_path(
            env_identifier
        )

    @property
    def model(self):
        return self._trainer.model

    def train(self) -> None:
        self._trainer.train()

    def save(self) -> None:
        saver = Sb3Saver(self._trainer.model, self._path_saveload)
        saver.save_model()

    def load(self, new_env: Env = None) -> BaseAlgorithm:
        return Sb3Loader(self._trainer.model, self._path_saveload).load_model(new_env)

    def test(self):
        model = self.load()
        PolicyTester.test_policy(model)

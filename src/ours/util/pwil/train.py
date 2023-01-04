import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
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
    def __init__(
        self,
        training_param: PwilParam,
        env_pwil_rewarded_and_identifier: tuple[Env, str],
    ):
        self._env_pwil_rewarded, env_identifier = env_pwil_rewarded_and_identifier

        self._reward_plot = self._make_reward_plot()
        self._path_saveload = PwilSaveLoadPathGenerator(training_param).get_plot_path(
            env_identifier
        )

    @property
    def reward_plot(self) -> np.ndarray:
        return self._reward_plot

    def _make_reward_plot(self) -> np.ndarray:
        plot = RewardPlotter.plot_reward(
            discriminator=None, env=self._env_pwil_rewarded
        )

        return plot

    def save_reward_plot(self, save_np: bool = True) -> None:
        im = Image.fromarray(self._reward_plot)

        save_path = str(self._path_saveload) + ".png"
        im.save(save_path)

        if save_np:
            self.save_reward_plot_np()

    def show_reward_plot(self) -> None:
        ax = plt.figure().subplots()
        ax.imshow(self._reward_plot)
        plt.show()

    def save_reward_plot_np(self) -> None:
        np.save(str(self._path_saveload), self._reward_plot)


class Sb3PwilTrainer(Trainer):
    def __init__(
        self,
        training_param: PwilParam,
        env_pwil_and_testing: tuple[Env, Env],
    ):
        self._training_param = training_param
        env_pwil_rewarded, env_raw_testing = env_pwil_and_testing

        self._model = PwilModelFactory(training_param, env_pwil_rewarded).model

        self._callback_list = CallbackListFactory(
            training_param, env_raw_testing
        ).callback_list

    @property
    def model(self):
        return self._model

    def train(self) -> None:
        self._model.learn(
            total_timesteps=self._training_param.n_steps_pwil_train,
            callback=self._callback_list,
        )


class Sb3PwilManager:
    def __init__(
        self,
        training_param: PwilParam,
        env_pwil_and_identifier: tuple[tuple[Env, Env], str],
    ):
        env_pwil_and_testing, env_identifier = env_pwil_and_identifier

        self._trainer = Sb3PwilTrainer(training_param, env_pwil_and_testing)

        self._path_saveload = PwilSaveLoadPathGenerator(training_param).get_model_path(
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


class PwilManager:
    def __init__(
        self,
        managers: tuple[Sb3PwilManager, TrajectoryManager, RewardPlotManager],
    ):
        (
            self._sb3_pwil_manager,
            self._trajectory_manager,
            self._reward_plot_manager,
        ) = managers

    def train_model(self) -> None:
        self._sb3_pwil_manager.train()

    def save_model(self) -> None:
        self._sb3_pwil_manager.save()

    def test_model(self) -> None:
        self._sb3_pwil_manager.test()

    def save_trajectory(self) -> None:
        self._trajectory_manager.save_trajectory()

    def load_trajectory(self) -> np.ndarray:
        return self._trajectory_manager.load_trajectory()

    def get_reward_plot(self) -> np.ndarray:
        return self._reward_plot_manager.reward_plot

    def train_and_save(self) -> None:
        self._sb3_pwil_manager.train()

        self._sb3_pwil_manager.save()
        self._trajectory_manager.save_trajectory()
        self._reward_plot_manager.save_reward_plot()


class PwilManagerFactory:
    def __init__(
        self,
        training_param: PwilParam,
        envs_and_identifier: tuple[tuple[Env, Env], str],
        trajectories: list[np.ndarray],
    ):
        (
            env_raw,
            env_raw_testing,
        ), env_identifier = envs_and_identifier

        env_pwil_rewarded = PwilEnvFactory(
            training_param, env_raw, trajectories
        ).env_pwil_rewarded

        self._sb3_pwil_manager = Sb3PwilManager(
            training_param,
            ((env_pwil_rewarded, env_raw_testing), env_identifier),
        )
        self._trajectory_manager = TrajectoryManager(
            (env_pwil_rewarded, env_identifier),
            (self._sb3_pwil_manager.model, training_param),
        )
        self._reward_plot_manager = RewardPlotManager(env_pwil_rewarded)

    @property
    def pwil_manager(self) -> PwilManager:
        return PwilManager(
            (
                self._sb3_pwil_manager,
                self._trajectory_manager,
                self._reward_plot_manager,
            )
        )

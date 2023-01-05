import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from gym import Env
from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.util.common.helper import RewardPlotter
from src.ours.util.common.param import PwilParam
from src.ours.util.common.pathprovider import PwilSaveLoadPathGenerator
from src.ours.util.expert.trajectory.manager import TrajectoryManager
from src.ours.util.pwil.sb3.manager import Sb3PwilManager
from src.upstream.env_utils import PWILReward


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

    def load_model(self) -> BaseAlgorithm:
        return self._sb3_pwil_manager.load()

    def save_trajectory(self) -> None:
        self._trajectory_manager.save()

    def load_trajectory(self) -> np.ndarray:
        return self._trajectory_manager.load()

    def save_reward_plot(self) -> None:
        self._reward_plot_manager.save_reward_plot()

    def show_reward_plot(self) -> None:
        self._reward_plot_manager.show_reward_plot()

    def get_reward_plot(self) -> np.ndarray:
        return self._reward_plot_manager.reward_plot

    def train_and_save(self) -> None:
        self._sb3_pwil_manager.train()

        self._sb3_pwil_manager.save()
        self._trajectory_manager.save()
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
        self._reward_plot_manager = RewardPlotManager(
            training_param, (env_pwil_rewarded, env_identifier)
        )

    @property
    def pwil_manager(self) -> PwilManager:
        return PwilManager(
            (
                self._sb3_pwil_manager,
                self._trajectory_manager,
                self._reward_plot_manager,
            )
        )

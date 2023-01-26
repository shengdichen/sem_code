import numpy as np
from gym import Env
from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.rl.pwil.param import PwilParam
from src.ours.rl.pwil.trajectory.manager import PwilTrajectoryManager
from src.ours.rl.pwil.rewardplot.manager import RewardPlotManager
from src.ours.rl.pwil.sb3.manager import PwilSb3Manager
from src.ours.rl.pwil.util.pwilenv import PwilEnvFactory


class PwilManager:
    def __init__(
        self,
        managers: tuple[RewardPlotManager, PwilSb3Manager, PwilTrajectoryManager],
    ):
        (
            self._reward_plot_manager,
            self._sb3_pwil_manager,
            self._trajectory_manager,
        ) = managers

    def save_reward_plot(self) -> None:
        self._reward_plot_manager.save()

    def show_reward_plot(self) -> None:
        self._reward_plot_manager.show_reward_plot()

    def get_reward_plot(self) -> np.ndarray:
        return self._reward_plot_manager.reward_plot

    def train_model(self) -> None:
        self._sb3_pwil_manager.train()

    def save_model(self) -> None:
        self._sb3_pwil_manager.save()

    def load_model(self) -> BaseAlgorithm:
        return self._sb3_pwil_manager.model

    def test_model(self) -> None:
        self._sb3_pwil_manager.test()

    def save_trajectory(self) -> None:
        self._trajectory_manager.save()

    def load_trajectory(self) -> np.ndarray:
        return self._trajectory_manager.load()

    def save_trajectory_stats_and_plot(self) -> None:
        self._trajectory_manager.save_stats()
        self._trajectory_manager.save_plot()

    def show_trajectory_plot(self) -> None:
        self._trajectory_manager.show_plot()


class PwilManagerFactoryConfig:
    use_raw_env_for_trajectory: bool = True


class PwilManagerFactory:
    def __init__(
        self,
        training_param: PwilParam,
        envs_and_identifier: tuple[tuple[Env, Env], str],
        trajectories: list[np.ndarray],
    ):
        (
            env_raw,
            env_eval,
        ), env_identifier = envs_and_identifier

        env_pwil_rewarded = PwilEnvFactory(
            training_param, env_raw, trajectories
        ).env_pwil_rewarded

        self._reward_plot_manager = RewardPlotManager(
            training_param, (env_pwil_rewarded, env_identifier)
        )
        self._sb3_pwil_manager = PwilSb3Manager(
            ((env_pwil_rewarded, env_eval), env_identifier),
            training_param,
        )
        if PwilManagerFactoryConfig.use_raw_env_for_trajectory:
            self._trajectory_manager = PwilTrajectoryManager(
                (env_raw, env_identifier),
                (self._sb3_pwil_manager.model, training_param),
            )
        else:
            self._trajectory_manager = PwilTrajectoryManager(
                (env_pwil_rewarded, env_identifier),
                (self._sb3_pwil_manager.model, training_param),
            )

    @property
    def pwil_manager(self) -> PwilManager:
        return PwilManager(
            (
                self._reward_plot_manager,
                self._sb3_pwil_manager,
                self._trajectory_manager,
            )
        )

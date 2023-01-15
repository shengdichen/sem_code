import numpy as np
from gym import Env

from src.ours.util.common.param import PwilParam
from src.ours.util.pwil.trajectory.manager import TrajectoryManager
from src.ours.util.pwil.rewardplot.manager import RewardPlotManager
from src.ours.util.pwil.sb3.manager import Sb3PwilManager
from src.ours.util.pwil.util.pwilenv import PwilEnvFactory


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
        self._trajectory_manager.save()

    def load_trajectory(self) -> np.ndarray:
        return self._trajectory_manager.load()

    def save_reward_plot(self) -> None:
        self._reward_plot_manager.save()

    def show_reward_plot(self) -> None:
        self._reward_plot_manager.show_reward_plot()

    def get_reward_plot(self) -> np.ndarray:
        return self._reward_plot_manager.reward_plot

    def train_and_save(self) -> None:
        self._sb3_pwil_manager.train()

        self._sb3_pwil_manager.save()
        self._trajectory_manager.save()
        self._reward_plot_manager.save()

    def save_trajectory_stats_and_plot(self) -> None:
        self._trajectory_manager.save_stats()
        self._trajectory_manager.save_plot()


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

        self._reward_plot_manager = RewardPlotManager(
            training_param, (env_pwil_rewarded, env_identifier)
        )
        self._sb3_pwil_manager = Sb3PwilManager(
            ((env_pwil_rewarded, env_raw_testing), env_identifier),
            training_param,
        )
        self._trajectory_manager = TrajectoryManager(
            (env_pwil_rewarded, env_identifier),
            (self._sb3_pwil_manager.model, training_param),
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

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import Env
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

from src.ours.env.creation import PathGenerator
from src.ours.eval.param import CommonParam
from src.ours.util.expert.saveload import ExpertSaveLoad
from src.ours.util.expert.trajectory import (
    TrajectoryGeneratorConfig,
    TrajectoryGenerator,
)
from src.ours.util.pathprovider import ExpertPathGenerator


class RewardPlotter:
    @staticmethod
    def plot_reward(discriminator, plot_value=False, env=None):
        # generate grid w/ two different targets:
        x = np.arange(0, 200)
        y = np.arange(0, 200)

        grid = np.array(np.meshgrid(x, y))
        point_list = np.reshape(grid, [2, -1]).T

        tgt_pos_1 = np.array([100, 100])
        tgt_pos_2 = np.array([190, 176])

        grid_pos1 = np.concatenate(
            [point_list, np.tile(tgt_pos_1, (point_list.shape[0], 1))], 1
        )
        grid_pos2 = np.concatenate(
            [point_list, np.tile(tgt_pos_2, (point_list.shape[0], 1))], 1
        )

        def dist_f(grid_pos):
            return -(
                np.sqrt(
                    (grid_pos[:, 0] - grid_pos[:, 2]) ** 2
                    + (grid_pos[:, 1] - grid_pos[:, 3]) ** 2
                )
            )

        r_gt_1 = dist_f(grid_pos1).reshape((200, 200))
        r_gt_2 = dist_f(grid_pos2).reshape((200, 200))

        # plot rewards for both goals
        if env is None:
            with torch.no_grad():
                r_1 = discriminator.get_reward(ob=torch.Tensor(grid_pos1), ac=None)
                r_2 = discriminator.get_reward(ob=torch.Tensor(grid_pos2), ac=None)
                if plot_value:
                    v_1 = discriminator.get_value(ob=torch.Tensor(grid_pos1), ac=None)
                    v_2 = discriminator.get_value(ob=torch.Tensor(grid_pos2), ac=None)

                r_1 = r_1.numpy()
                r_2 = r_2.numpy()

        else:
            r1 = []
            r2 = []
            for pos in grid_pos1:
                r1.append(env.pwil.compute_reward(pos))
            for pos in grid_pos2:
                r2.append(env.pwil.compute_reward(pos))
            r_1 = np.stack(r1)
            r_2 = np.stack(r2)

        title = "reward"
        if plot_value:
            r_1 = v_1.numpy()
            r_2 = v_2.numpy()
            title = "value"

        fig = plt.gcf()
        fig.set_size_inches(24, 5)

        plt.subplot(141)
        plt.pcolor(x, y, r_1.reshape((200, 200)))
        plt.colorbar()
        plt.scatter(tgt_pos_1[0], tgt_pos_1[1], c="r")
        plt.title(title)
        plt.axis("equal")

        plt.subplot(142)
        plt.pcolor(x, y, r_gt_1)
        plt.colorbar()
        plt.scatter(tgt_pos_1[0], tgt_pos_1[1], c="r")
        plt.title(title)
        plt.axis("equal")

        plt.subplot(143)
        plt.pcolor(x, y, r_2.reshape((200, 200)))
        plt.colorbar()
        plt.scatter(tgt_pos_2[0], tgt_pos_2[1], c="r")
        plt.title(title)
        plt.axis("equal")

        plt.subplot(144)
        plt.pcolor(x, y, r_gt_2)
        plt.colorbar()
        plt.title(title)
        plt.scatter(tgt_pos_2[0], tgt_pos_2[1], c="r")
        plt.axis("equal")

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image_from_plot


class TqdmCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.progress_bar = None

    def _on_training_start(self):
        self.progress_bar = tqdm(total=self.locals["total_timesteps"])

    def _on_step(self):
        self.progress_bar.update(1)
        return True

    def _on_training_end(self):
        self.progress_bar.close()
        self.progress_bar = None


class RewardCheckpointCallback(BaseCallback):
    def __init__(self, discriminator, verbose=0, plot_value=False, log_path=None):
        super(RewardCheckpointCallback, self).__init__(verbose)
        self.discriminator = discriminator
        self.log_path = log_path
        self.plot_value = plot_value
        self.plot_list_reward = []
        self.plot_list_value = []

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        self.plot_list_reward.append(RewardPlotter.plot_reward(self.discriminator))
        if self.plot_value:
            self.plot_list_value.append(
                RewardPlotter.plot_reward(self.discriminator, plot_value=True)
            )
        torch.save(
            self.discriminator.state_dict(), os.path.join(self.log_path, "best_disc.th")
        )
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass


class ExpertManager:
    def __init__(
        self,
        env_model: tuple[Env, Any],
        training_param: CommonParam,
        expert_manager_param=TrajectoryGeneratorConfig(),
    ):
        self._expert_generator = TrajectoryGenerator(env_model, expert_manager_param)
        self._training_param = training_param

    def save_expert_traj(self, filename="exp"):
        expert_traj = self._expert_generator.get_trajectory()
        path_saveload = ExpertPathGenerator(self._training_param).get_path(filename)

        ExpertSaveLoad(path_saveload).save(expert_traj)

    def load_one_demo(self, filename: str):
        path_saveload = ExpertPathGenerator(self._training_param).get_path(filename)
        return ExpertSaveLoad(path_saveload).load()

    def load_default_demos(self):
        expert_demos = []
        for env_config in [
            {"n_targets": 2, "shift_x": 0, "shift_y": 0},
            {"n_targets": 2, "shift_x": 0, "shift_y": 50},
            {"n_targets": 2, "shift_x": 50, "shift_y": 0},
        ]:
            filename = PathGenerator(env_config).get_filename_from_shift_values()
            demo = self.load_one_demo(filename)
            expert_demos.append(demo)

        return expert_demos


class Plotter:
    def __init__(self):
        pass  # intentionally left empty

    @staticmethod
    # TODO:
    #   name of demo-file should be a parameter, not hard-coded!
    def plot_experts(n_timesteps=3e5, hist=True):
        demo1 = Plotter.plot_traj(
            "demos/exp_0_0" + str(n_timesteps) + "_expert_traj.npy"
        )
        demo2 = Plotter.plot_traj(
            "demos/exp_50_0" + str(n_timesteps) + "_expert_traj.npy"
        )
        demo3 = Plotter.plot_traj(
            "demos/exp_0_50" + str(n_timesteps) + "_expert_traj.npy"
        )

        plt.figure(figsize=[15, 5])

        plt.subplot(131)
        x, y, bins = Plotter.get_hist_data(demo1)
        x_tgt = demo1[:, 2]
        y_tgt = demo1[:, 3]
        if hist:
            plt.hist2d(x, y, bins)
        else:
            plt.plot(x, y, "m-", alpha=0.3)
        plt.scatter(x_tgt, y_tgt, c="r")

        plt.subplot(132)
        x, y, bins = Plotter.get_hist_data(demo2)
        x_tgt = demo2[:, 2]
        y_tgt = demo2[:, 3]
        if hist:
            plt.hist2d(x, y, bins)
        else:
            plt.plot(x, y, "m-", alpha=0.3)
        plt.scatter(x_tgt, y_tgt, c="r")

        plt.subplot(133)
        x, y, bins = Plotter.get_hist_data(demo3)
        x_tgt = demo3[:, 2]
        y_tgt = demo3[:, 3]
        if hist:
            plt.hist2d(x, y, bins)
        else:
            plt.plot(x, y, "m-", alpha=0.3)
        plt.scatter(x_tgt, y_tgt, c="r")

    @staticmethod
    def plot_traj(fname, plot=False):
        demo = np.load(fname)

        # state visitation
        if plot:
            plt.figure()
            plt.hist2d(x, y, bins=[x_bins, y_bins])

            plt.figure()
            # action distribution
            plt.hist(demo[:, 4])

        # reward stats
        num_episodes = np.sum(demo[:, -1])
        rew_avg = np.mean(demo[:, -2])
        rew_std = np.std(demo[:, -2])
        rew_min = np.min(demo[:, -2])
        rew_max = np.max(demo[:, -2])

        ep_rew_list = []
        ep_rew = 0
        for sard in demo:
            ep_rew += sard[-2]
            if sard[-1] == 1:
                ep_rew_list.append(ep_rew)
                # print("episode_reward", ep_rew)
                ep_rew = 0

        ep_rew_avg = np.mean(ep_rew_list)
        ep_rew_std = np.std(ep_rew_list)
        ep_rew_min = np.min(ep_rew_list)
        ep_rew_max = np.max(ep_rew_list)

        print("Demo file stats")
        print(fname)
        print("-------------")
        print("Number of episodes: ", num_episodes)
        print("Reward stats: ", rew_avg, " +- ", rew_std)
        print("Reward min / max", rew_min, " / ", rew_max)
        print("Episode reward stats: ", ep_rew_avg, " +- ", ep_rew_std)
        print("Episode reward min / max", ep_rew_min, " / ", ep_rew_max)
        print("-------------")

        return demo

    @staticmethod
    def get_hist_data(demo, nr=40, canvas_size=200):
        x = demo[:, 0]
        y = demo[:, 1]
        x_bins = np.linspace(0, canvas_size, nr)
        y_bins = np.linspace(0, canvas_size, nr)

        return x, y, [x_bins, y_bins]

    @staticmethod
    def get_np_min_max(vec: np.ndarray):
        return np.min(vec), np.max(vec)

    @staticmethod
    def get_np_min_max_x_y(vec_x: np.ndarray, vec_y: np.ndarray):
        return Plotter.get_np_min_max(vec_x), Plotter.get_np_min_max(vec_y)

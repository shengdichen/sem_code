import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

from src.ours.util.expert.analyzer.plotter import TrajectoryInspector


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


class TrajectoriesPlotter:
    def __init__(self, trajectories: list[np.ndarray]):
        self._trajectories = trajectories
        self._n_trajectories = len(trajectories)

        self._figures = self._get_configured_figures()

    def _get_configured_figures(self):
        figure = plt.figure(figsize=[15, 5])
        subfigures = figure.subfigures(1, self._n_trajectories)

        return subfigures

    @staticmethod
    def _show_figures():
        plt.show()

    def plot_experts(self, plot_hist=True):
        for trajectory in self._trajectories:
            TrajectoryInspector(trajectory).display_stats()

        for trajectory, subfigure in zip(self._trajectories, self._figures):
            axs = subfigure.subplots(1, 2)
            TrajectoryInspector(trajectory).plot_agent_and_target(axs, plot_hist)

        self._show_figures()

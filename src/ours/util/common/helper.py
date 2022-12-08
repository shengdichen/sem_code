import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm


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

    def plot_experts(self, plot_hist=True):
        for trajectory in self._trajectories:
            TrajectoryInspector(trajectory).display_stats()

        figure = plt.figure(figsize=[15, 5])
        subfigures = figure.subfigures(1, self._n_trajectories)

        for trajectory, subfigure in zip(self._trajectories, subfigures):
            axs = subfigure.subplots(1, 2)
            TrajectoryInspector(trajectory).plot_agent_and_target(axs, plot_hist)

        plt.show()


class TrajectoryInspector:
    def __init__(self, trajectory: np.ndarray):
        self._trajectory = trajectory
        self._trajectory_interpreter = TrajectoryInterpreter(self._trajectory)

    def plot_agent_and_target(
        self, axs: tuple[plt.Axes, plt.Axes], plot_hist: bool
    ) -> None:
        if plot_hist:
            self._plot_hist(axs[0])
        else:
            self._plot_agent(axs[0])

        self._plot_target(axs[1])

    def _plot_agent(self, ax: plt.Axes) -> None:
        agent_pos_x, agent_pos_y, __ = self.get_hist_data()
        ax.plot(agent_pos_x, agent_pos_y, "m-", alpha=0.3)

    def _plot_target(self, ax: plt.Axes) -> None:
        target_pos_x, target_pos_y = self._trajectory_interpreter.target_pos
        ax.scatter(target_pos_x, target_pos_y, c="r")

    def display_stats(self) -> None:
        num_episodes = self._trajectory_interpreter.get_num_episodes()

        (
            rew_avg,
            rew_std,
            rew_min,
            rew_max,
        ) = self._trajectory_interpreter.get_reward_stats()

        (
            ep_rew_avg,
            ep_rew_std,
            ep_rew_min,
            ep_rew_max,
        ) = self._trajectory_interpreter.get_episode_reward_stats()

        print("Demo file stats")
        print("-------------")
        print("Number of episodes: ", num_episodes)
        print("Reward stats: ", rew_avg, " +- ", rew_std)
        print("Reward min / max", rew_min, " / ", rew_max)
        print("Episode reward stats: ", ep_rew_avg, " +- ", ep_rew_std)
        print("Episode reward min / max", ep_rew_min, " / ", ep_rew_max)
        print("-------------")

    def _plot_hist_and_action(self) -> None:
        # state visitation
        __, axs = plt.subplots(1, 2)

        self._plot_hist(axs[0])
        self._plot_action(axs[1])

        plt.show()

    def _plot_hist(self, ax: plt.Axes) -> None:
        x, y, [x_bins, y_bins] = self.get_hist_data()
        ax.hist2d(x, y, bins=[x_bins, y_bins])

    def _plot_action(self, ax: plt.Axes) -> None:
        # action distribution
        ax.hist(self._trajectory_interpreter.action)

    def get_hist_data(self, nr=40, canvas_size=200):
        agent_pos_x, agent_pos_y = self._trajectory_interpreter.agent_pos
        x_bins = np.linspace(0, canvas_size, nr)
        y_bins = np.linspace(0, canvas_size, nr)

        return agent_pos_x, agent_pos_y, [x_bins, y_bins]


class TrajectoryInterpreter:
    def __init__(self, trajectory: np.ndarray):
        self._trajectory = trajectory

    @property
    def agent_pos(self) -> tuple[np.ndarray, np.ndarray]:
        return self._trajectory[:, 0], self._trajectory[:, 1]

    @property
    def target_pos(self) -> tuple[np.ndarray, np.ndarray]:
        return self._trajectory[:, 2], self._trajectory[:, 3]

    @property
    def action(self) -> np.ndarray:
        return self._trajectory[:, 4]

    @property
    def reward(self) -> np.ndarray:
        return self._trajectory[:, 5]

    @property
    def done(self):
        return self._trajectory[:, 6]

    def get_num_episodes(self) -> int:
        return int(np.sum(self.done))

    def get_reward_stats(self) -> tuple[float, float, float, float]:
        # reward stats
        rew_avg, rew_std = TrajectoryInterpreter._get_avg_std(self.reward)
        rew_min, rew_max = MinMaxUtil.get_np_min_max(self.reward)

        return rew_avg, rew_std, rew_min, rew_max

    def _get_episode_reward_list(self) -> np.ndarray:
        ep_rew_list = []
        ep_rew = 0
        for sard in self._trajectory:
            ep_rew += sard[-2]
            if sard[-1] == 1:
                ep_rew_list.append(ep_rew)
                # print("episode_reward", ep_rew)
                ep_rew = 0

        return np.array(ep_rew_list)

    def get_episode_reward_stats(self) -> tuple[float, float, float, float]:
        ep_rew_list = self._get_episode_reward_list()

        ep_rew_avg, ep_rew_std = TrajectoryInterpreter._get_avg_std(ep_rew_list)
        ep_rew_min, ep_rew_max = MinMaxUtil.get_np_min_max(ep_rew_list)

        return ep_rew_avg, ep_rew_std, ep_rew_min, ep_rew_max

    @staticmethod
    def _get_avg_std(data: np.ndarray) -> tuple[float, float]:
        return float(np.mean(data)), float(np.std(data))


class MinMaxUtil:
    @staticmethod
    def get_np_min_max(vec: np.ndarray) -> tuple[float, float]:
        return float(np.min(vec)), float(np.max(vec))

    @staticmethod
    def get_np_min_max_x_y(vec_x: np.ndarray, vec_y: np.ndarray):
        return MinMaxUtil.get_np_min_max(vec_x), MinMaxUtil.get_np_min_max(vec_y)

import numpy as np
from matplotlib import pyplot as plt

from src.ours.util.expert.analyzer.stats import TrajectoryStats


class TrajectoryInspector:
    def __init__(self, trajectory: np.ndarray):
        self._trajectory_stats = TrajectoryStats(trajectory)

        self._bins_hist = self._make_bins_hist()

    @staticmethod
    def _make_bins_hist(nr=40, canvas_size=200) -> tuple[np.ndarray, np.ndarray]:
        x_bins = np.linspace(0, canvas_size, nr)
        y_bins = np.linspace(0, canvas_size, nr)

        return x_bins, y_bins

    def plot_agent_and_target(
        self, axs: tuple[plt.Axes, plt.Axes], plot_agent_with_hist: bool
    ) -> None:
        if plot_agent_with_hist:
            self._plot_agent_hist(axs[0])
        else:
            self._plot_agent_direct(axs[0])

        self._plot_target(axs[1])

    def _plot_agent_hist(self, ax: plt.Axes) -> None:
        agent_pos_x, agent_pos_y = self._trajectory_stats.agent_pos
        ax.hist2d(agent_pos_x, agent_pos_y, bins=self._bins_hist)

    def _plot_agent_direct(self, ax: plt.Axes) -> None:
        agent_pos_x, agent_pos_y = self._trajectory_stats.agent_pos
        ax.plot(agent_pos_x, agent_pos_y, "m-", alpha=0.3)

    def _plot_target(self, ax: plt.Axes) -> None:
        target_pos_x, target_pos_y = self._trajectory_stats.target_pos
        ax.scatter(target_pos_x, target_pos_y, c="r")

    def _plot_hist_and_action(self) -> None:
        __, axs = plt.subplots(1, 2)

        self._plot_agent_hist(axs[0])
        self._plot_action(axs[1])

        plt.show()

    def _plot_action(self, ax: plt.Axes) -> None:
        ax.hist(self._trajectory_stats.action)

    def display_stats(self) -> None:
        self._trajectory_stats.display_stats()

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from src.ours.util.expert.analyzer.stats import TrajectoryStats


class TrajectoryPlotter:
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
            self.plot_agent_hist(axs[0])
        else:
            self.plot_agent_direct(axs[0])

        self.plot_target(axs[1])

    def plot_hist_and_action(self) -> None:
        __, axs = plt.subplots(1, 2)

        self.plot_agent_hist(axs[0])
        self.plot_action(axs[1])

        plt.show()

    def plot_agent_hist(self, ax: plt.Axes) -> None:
        agent_pos_x, agent_pos_y = self._trajectory_stats.agent_pos
        ax.hist2d(agent_pos_x, agent_pos_y, bins=self._bins_hist)

    def plot_agent_direct(self, ax: plt.Axes) -> None:
        agent_pos_x, agent_pos_y = self._trajectory_stats.agent_pos
        ax.plot(agent_pos_x, agent_pos_y, "m-", alpha=0.3)

    def plot_target(self, ax: plt.Axes) -> None:
        target_pos_x, target_pos_y = self._trajectory_stats.target_pos
        ax.scatter(target_pos_x, target_pos_y, c="r")

    def plot_action(self, ax: plt.Axes) -> None:
        ax.hist(self._trajectory_stats.action)

    def display_stats(self) -> None:
        self._trajectory_stats.display_stats()


class TrajectoryMultiPlotter:
    def __init__(
        self,
        trajectory_plotter: TrajectoryPlotter,
        figure: matplotlib.figure.Figure | matplotlib.figure.SubFigure,
    ):
        self._trajectory_plotter = trajectory_plotter
        self._figure = figure

    def plot_agent_and_target(self, plot_agent_with_hist: bool) -> None:
        axs = self._figure.subplots(1, 2)

        if plot_agent_with_hist:
            self._trajectory_plotter.plot_agent_hist(axs[0])
        else:
            self._trajectory_plotter.plot_agent_direct(axs[0])

        self._trajectory_plotter.plot_target(axs[1])

    def plot_hist_and_action(self) -> None:
        axs = self._figure.subplots(1, 2)

        self._trajectory_plotter.plot_agent_hist(axs[0])
        self._trajectory_plotter.plot_action(axs[1])

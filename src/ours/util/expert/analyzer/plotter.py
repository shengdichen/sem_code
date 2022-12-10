import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from src.ours.util.expert.analyzer.stats import TrajectoryStats


class TrajectoryPlotter:
    def __init__(self, trajectory: np.ndarray, ax: plt.Axes):
        self._trajectory_stats = TrajectoryStats(trajectory)

        self._ax = ax
        self._bins_hist = self._make_bins_hist()

    @staticmethod
    def _make_bins_hist(nr=40, canvas_size=200) -> tuple[np.ndarray, np.ndarray]:
        x_bins = np.linspace(0, canvas_size, nr)
        y_bins = np.linspace(0, canvas_size, nr)

        return x_bins, y_bins

    def plot_agent(self, plot_hist: bool) -> None:
        if plot_hist:
            self._plot_agent_hist()
        else:
            self._plot_agent_direct()

    def _plot_agent_hist(self) -> None:
        agent_pos_x, agent_pos_y = self._trajectory_stats.agent_pos
        self._ax.hist2d(agent_pos_x, agent_pos_y, bins=self._bins_hist)

    def _plot_agent_direct(self) -> None:
        agent_pos_x, agent_pos_y = self._trajectory_stats.agent_pos
        self._ax.plot(agent_pos_x, agent_pos_y, "m-", alpha=0.3)

    def plot_target(self) -> None:
        target_pos_x, target_pos_y = self._trajectory_stats.target_pos
        self._ax.scatter(target_pos_x, target_pos_y, c="r")

    def plot_action(self) -> None:
        self._ax.hist(self._trajectory_stats.action)


class TrajectoryMultiPlotter:
    def __init__(
        self,
        trajectory: np.ndarray,
        figure: matplotlib.figure.Figure | matplotlib.figure.SubFigure,
    ):
        self._trajectory = trajectory
        self._figure = figure

    def plot_agent_and_target(self, plot_agent_as_hist: bool) -> None:
        axs = self._figure.subplots(1, 2)

        TrajectoryPlotter(self._trajectory, axs[0]).plot_agent(plot_agent_as_hist)
        TrajectoryPlotter(self._trajectory, axs[1]).plot_target()

    def plot_agent_and_action(self, plot_agent_as_hist: bool) -> None:
        axs = self._figure.subplots(1, 2)

        TrajectoryPlotter(self._trajectory, axs[0]).plot_agent(plot_agent_as_hist)
        TrajectoryPlotter(self._trajectory, axs[1]).plot_action()

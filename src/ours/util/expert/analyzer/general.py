import numpy as np
from matplotlib import pyplot as plt

from src.ours.util.expert.analyzer.plotter import TrajectoryPlotter
from src.ours.util.expert.analyzer.stats import TrajectoryStats


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

    def analyze(self, plot_hist=True):
        for trajectory in self._trajectories:
            TrajectoryStats(trajectory).display_stats()

        for trajectory, subfigure in zip(self._trajectories, self._figures):
            axs = subfigure.subplots(1, 2)
            TrajectoryPlotter(trajectory).plot_agent_and_target(axs, plot_hist)

        self._show_figures()

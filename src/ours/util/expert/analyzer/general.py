import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from src.ours.util.expert.analyzer.plotter import (
    TrajectoryPlotter,
    TrajectoryMultiPlotter,
)
from src.ours.util.expert.analyzer.stats import TrajectoryStats


class TrajectoriesAnalyzer:
    def __init__(self, trajectories: list[np.ndarray]):
        self._trajectories = trajectories

        self._figures = self._get_configured_figures()

    def _get_configured_figures(self) -> list[matplotlib.figure.SubFigure]:
        figure = plt.figure(figsize=[15, 5])
        subfigures = figure.subfigures(1, len(self._trajectories))

        return subfigures

    @staticmethod
    def _show_figures() -> None:
        plt.show()

    def analyze(self, plot_agent_as_hist: bool = True) -> None:
        for trajectory in self._trajectories:
            TrajectoryStats(trajectory).display_stats()

        for trajectory, subfigure in zip(self._trajectories, self._figures):
            TrajectoryMultiPlotter(
                TrajectoryPlotter(trajectory), subfigure
            ).plot_agent_and_target(plot_agent_as_hist)

        self._show_figures()

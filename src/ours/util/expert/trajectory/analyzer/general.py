from abc import ABC, abstractmethod

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from src.ours.util.expert.trajectory.analyzer.plot import TrajectoryPlot
from src.ours.util.expert.trajectory.analyzer.stats import TrajectoryStats
from src.ours.util.expert.trajectory.analyzer.util import MplUtil


class TrajectoriesStats:
    def __init__(self, trajectories: list[np.ndarray]):
        self._trajectories = trajectories

        self._trajectories_stats = [
            TrajectoryStats(trajectory) for trajectory in self._trajectories
        ]

    def show_stats(self) -> None:
        for trajectory_stats in self._trajectories_stats:
            print(trajectory_stats.get_stats())


class TrajectoriesPlotBase(ABC):
    def __init__(self, trajectories: list[np.ndarray]):
        self._trajectories = trajectories

    def analyze(self, plot_agent_as_hist: bool = True) -> None:
        for trajectory_plotter in self._get_trajectories_plotter():
            trajectory_plotter.plot_agent_and_target(plot_agent_as_hist)

        MplUtil.show_figures()

    def _get_trajectories_plotter(self) -> list[TrajectoryPlot]:
        return [
            TrajectoryPlot(trajectory, figure)
            for trajectory, figure in zip(
                self._trajectories, self._get_configured_figures()
            )
        ]

    @abstractmethod
    def _get_configured_figures(self) -> list[matplotlib.figure.FigureBase]:
        pass


class TrajectoriesAnalyzerParallel(TrajectoriesPlotBase):
    def __init__(self, trajectories: list[np.ndarray]):
        super().__init__(trajectories)

        self._figure = plt.figure(figsize=[15, 5])

    def _get_configured_figures(
        self,
    ) -> list[matplotlib.figure.Figure] | list[matplotlib.figure.SubFigure]:
        return MplUtil(len(self._trajectories)).get_horizontal_figures(self._figure)


class TrajectoriesAnalyzerSeparate(TrajectoriesPlotBase):
    def __init__(self, trajectories: list[np.ndarray]):
        super().__init__(trajectories)

        self._figures = MplUtil(len(self._trajectories)).get_separate_figures()

    def _get_configured_figures(self) -> list[matplotlib.figure.Figure]:
        return self._figures

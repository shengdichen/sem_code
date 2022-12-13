from abc import ABC, abstractmethod

import matplotlib
import numpy as np

from src.ours.util.expert.trajectory.analyzer.plotter import TrajectoryPlotter
from src.ours.util.expert.trajectory.analyzer.stats import TrajectoryStats
from src.ours.util.expert.trajectory.analyzer.util import MplUtil


class TrajectoriesAnalyzerBase(ABC):
    def __init__(self, trajectories: list[np.ndarray]):
        self._trajectories = trajectories

        self._trajectories_stats = [
            TrajectoryStats(trajectory) for trajectory in self._trajectories
        ]

    def analyze(self, plot_agent_as_hist: bool = True) -> None:
        for trajectory_stats in self._trajectories_stats:
            trajectory_stats.display_stats()

        for trajectory_plotter in self._get_trajectories_plotter():
            trajectory_plotter.plot_agent_and_target(plot_agent_as_hist)

        MplUtil.show_figures()

    def _get_trajectories_plotter(self) -> list[TrajectoryPlotter]:
        return [
            TrajectoryPlotter(trajectory, figure)
            for trajectory, figure in zip(
                self._trajectories, self._get_configured_figures()
            )
        ]

    @abstractmethod
    def _get_configured_figures(self) -> list[matplotlib.figure.FigureBase]:
        pass


class TrajectoriesAnalyzer(TrajectoriesAnalyzerBase):
    def __init__(self, trajectories: list[np.ndarray]):
        super().__init__(trajectories)

    def _get_configured_figures(
        self,
    ) -> list[matplotlib.figure.Figure] | list[matplotlib.figure.SubFigure]:
        return MplUtil(len(self._trajectories)).get_parallel_figures()


class TrajectoriesAnalyzerSeparate(TrajectoriesAnalyzerBase):
    def __init__(self, trajectories: list[np.ndarray]):
        super().__init__(trajectories)

    def _get_configured_figures(self) -> list[matplotlib.figure.Figure]:
        return MplUtil(len(self._trajectories)).get_separate_figures()

from abc import ABC, abstractmethod

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from src.ours.util.expert.trajectory.analyzer.plot.single import TrajectoryPlot
from src.ours.util.expert.trajectory.analyzer.util import MplUtil


class TrajectoriesPlot(ABC):
    def __init__(self, trajectories: list[np.ndarray]):
        self._trajectories = trajectories

    def show_plot(self, plot_agent_as_hist: bool = True) -> None:
        for trajectory_plot in self._get_trajectories_plot():
            trajectory_plot.plot_agent_and_target(plot_agent_as_hist)

        MplUtil.show_figures()

    def _get_trajectories_plot(self) -> list[TrajectoryPlot]:
        return [
            TrajectoryPlot(trajectory, figure)
            for trajectory, figure in zip(
                self._trajectories, self._get_configured_figures()
            )
        ]

    @abstractmethod
    def _get_configured_figures(self) -> list[matplotlib.figure.FigureBase]:
        pass


class ParallelTrajectoriesPlot(TrajectoriesPlot):
    def __init__(self, trajectories: list[np.ndarray]):
        super().__init__(trajectories)

        self._figure = plt.figure(figsize=[15, 5])

    def _get_configured_figures(
        self,
    ) -> list[matplotlib.figure.SubFigure]:
        return MplUtil(len(self._trajectories)).get_horizontal_figures(self._figure)


class TrajectoriesPlotSeparate(TrajectoriesPlot):
    def __init__(self, trajectories: list[np.ndarray]):
        super().__init__(trajectories)

        self._figures = MplUtil(len(self._trajectories)).get_separate_figures()

    def _get_configured_figures(self) -> list[matplotlib.figure.Figure]:
        return self._figures

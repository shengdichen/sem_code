import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from src.ours.util.expert.trajectory.analyzer.plotter import TrajectoryPlotter
from src.ours.util.expert.trajectory.analyzer.stats import TrajectoryStats


class MplUtil:
    def __init__(self, n_trajectories: int):
        self._n_trajectories = n_trajectories

    def get_parallel_figures(self):
        figure = plt.figure(figsize=[15, 5])

        if self._n_trajectories == 1:
            return [figure]
        else:
            return figure.subfigures(1, self._n_trajectories)

    def get_separate_figures(self):
        return [plt.figure(figsize=[15, 5]) for __ in range(self._n_trajectories)]

    @staticmethod
    def _show_figures() -> None:
        plt.show()


class TrajectoriesAnalyzerBase:
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

        MplUtil._show_figures()

    def _get_trajectories_plotter(self) -> list[TrajectoryPlotter]:
        return [
            TrajectoryPlotter(trajectory, figure)
            for trajectory, figure in zip(
                self._trajectories, self._get_configured_figures()
            )
        ]

    def _get_configured_figures(self) -> list[matplotlib.figure.SubFigure]:
        pass


class TrajectoriesAnalyzer(TrajectoriesAnalyzerBase):
    def __init__(self, trajectories: list[np.ndarray]):
        super().__init__(trajectories)

    def _get_configured_figures(self) -> list[matplotlib.figure.SubFigure]:
        return MplUtil(len(self._trajectories)).get_parallel_figures()


class TrajectoriesAnalyzerSeparate(TrajectoriesAnalyzerBase):
    def __init__(self, trajectories: list[np.ndarray]):
        super().__init__(trajectories)

    def _get_configured_figures(self) -> list[matplotlib.figure.SubFigure]:
        return MplUtil(len(self._trajectories)).get_separate_figures()

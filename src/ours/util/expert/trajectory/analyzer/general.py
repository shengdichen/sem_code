import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from src.ours.util.expert.trajectory.analyzer.plotter import TrajectoryPlotter
from src.ours.util.expert.trajectory.analyzer.stats import TrajectoryStats


class TrajectoriesAnalyzerBase:
    def __init__(self, trajectories: list[np.ndarray]):
        self._trajectories = trajectories

        self._trajectories_stats = [
            TrajectoryStats(trajectory) for trajectory in self._trajectories
        ]

    def _get_trajectories_plotter(self) -> list[TrajectoryPlotter]:
        return [
            TrajectoryPlotter(trajectory, figure)
            for trajectory, figure in zip(
                self._trajectories, self._get_configured_figures()
            )
        ]

    def _get_configured_figures(self) -> list[matplotlib.figure.SubFigure]:
        pass

    def analyze(self, plot_agent_as_hist: bool = True) -> None:
        for trajectory_stats in self._trajectories_stats:
            trajectory_stats.display_stats()

        for trajectory_plotter in self._get_trajectories_plotter():
            trajectory_plotter.plot_agent_and_target(plot_agent_as_hist)

        self._show_figures()

    @staticmethod
    def _show_figures() -> None:
        plt.show()


class TrajectoriesAnalyzer(TrajectoriesAnalyzerBase):
    def __init__(self, trajectories: list[np.ndarray]):
        super().__init__(trajectories)

    def _get_configured_figures(self) -> list[matplotlib.figure.SubFigure]:
        figure = plt.figure(figsize=[15, 5])

        n_trajectories = len(self._trajectories)
        if n_trajectories == 1:
            return [figure]
        else:
            return figure.subfigures(1, n_trajectories)


class TrajectoriesAnalyzerSeparate(TrajectoriesAnalyzerBase):
    def __init__(self, trajectories: list[np.ndarray]):
        super().__init__(trajectories)

    def _get_configured_figures(self) -> list[matplotlib.figure.SubFigure]:
        return [plt.figure(figsize=[15, 5]) for __ in range(len(self._trajectories))]

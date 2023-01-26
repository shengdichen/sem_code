from abc import ABC, abstractmethod

import matplotlib
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from src.ours.rl.common.trajectory.analyzer.plot.single import TrajectoryPlot
from src.ours.rl.common.trajectory.analyzer.stats.multi import TrajectoriesStats
from src.ours.rl.common.trajectory.analyzer.util import MplUtil
from src.ours.rl.pwil.param import PwilParam


class Selector:
    def __init__(self, trajectories: list[np.ndarray], pwil_params: list[PwilParam]):
        self._trajectories = trajectories
        self._params = pwil_params

    @property
    def trajectories(self) -> list[np.ndarray]:
        return self._trajectories

    @property
    def params(self) -> list[PwilParam]:
        return self._params

    def select_by_trajectory_num(self, candidates: list[int]) -> "Selector":
        self._trajectories, self._params = self._select_by(
            self._is_selected_by_trajectory_num, candidates
        )
        return self

    def select_by_n_demos(self, candidates: list[int]) -> "Selector":
        self._trajectories, self._params = self._select_by(
            self._is_selected_by_n_demos, candidates
        )
        return self

    def select_by_subsampling(self, candidates: list[int]) -> "Selector":
        self._trajectories, self._params = self._select_by(
            self._is_selected_by_subsampling, candidates
        )
        return self

    def _select_by(
        self, _is_selected_function, candidates: list[int]
    ) -> tuple[list[np.ndarray], list[PwilParam]]:
        selected_trajectories, selected_params = [], []
        for trajectory, param in zip(self._trajectories, self._params):
            if _is_selected_function(param, candidates):
                selected_trajectories.append(trajectory)
                selected_params.append(param)

        return selected_trajectories, selected_params

    @staticmethod
    def _is_selected_by_trajectory_num(param: PwilParam, candidates: list[int]) -> bool:
        for candidate in candidates:
            if param.trajectory_num == candidate:
                return True
        return False

    @staticmethod
    def _is_selected_by_n_demos(param: PwilParam, candidates: list[int]) -> bool:
        for candidate in candidates:
            if param.pwil_training_param["n_demos"] == candidate:
                return True
        return False

    @staticmethod
    def _is_selected_by_subsampling(param: PwilParam, candidates: list[int]) -> bool:
        for candidate in candidates:
            if param.pwil_training_param["subsampling"] == candidate:
                return True
        return False


class TrajectoriesComparisonPlot:
    def __init__(self, trajectories: list[np.ndarray], pwil_params: list[PwilParam]):
        self._trajectories = trajectories
        self._params = pwil_params

        self._demo_id_to_demo_quality = {
            0: "optimal",
            1: "mixed",
            2: "mixed",
            3: "mixed",
            4: "distant",
            5: "distant",
            6: "distant",
        }

    def compare_optimal(self, plot_together: bool = True):
        selection_optimal_one = (
            Selector(self._trajectories, self._params)
            .select_by_trajectory_num([0])
            .select_by_n_demos([1])
        )
        selection_optimal_five = (
            Selector(self._trajectories, self._params)
            .select_by_trajectory_num([0])
            .select_by_n_demos([5])
        )
        selection_optimal_ten = (
            Selector(self._trajectories, self._params)
            .select_by_trajectory_num([0])
            .select_by_n_demos([10])
        )

        if plot_together:
            self._plot_selections_together(
                plt.figure().subplots(),
                [selection_optimal_one, selection_optimal_five, selection_optimal_ten],
            )
        else:
            axes = plt.figure().subplots(1, 3)
            self._plot_selections_separate(
                axes,
                [selection_optimal_one, selection_optimal_five, selection_optimal_ten],
                "subsampling",
            )

        plt.show()

    def _plot_selections_together(self, ax: mpl.axes.Axes, selections: list[Selector]):
        for selection in selections:
            subsamplings = [
                param.pwil_training_param["subsampling"] for param in selection.params
            ]

            n_demos = selection.params[0].pwil_training_param["n_demos"]
            ax.plot(
                subsamplings,
                TrajectoriesStats(selection.trajectories).rewards_avg,
                "x--",
                label="num-demos: {0}".format(n_demos),
            )

        demo_id = selections[0].params[0].trajectory_num
        demo_type = self._demo_id_to_demo_quality[demo_id]
        ax.set_title(
            "[demo-type]-[n-traj]: {0}-{1}".format(demo_type, "[1 | 5 | 10]"),
        )
        ax.legend()

    def _plot_selections_separate(
        self, axes: list[mpl.axes.Axes], selections: list[Selector], variant: str
    ):
        for ax, selection in zip(axes, selections):
            self._plot_selection(ax, selection, variant)

    def _plot_selection(self, ax: mpl.axes.Axes, selection: Selector, variant: str):
        if variant == "subsampling":
            subsamplings = [
                param.pwil_training_param["subsampling"] for param in selection.params
            ]

            ax.plot(
                subsamplings,
                TrajectoriesStats(selection.trajectories).rewards_avg,
                "x--",
            )
            ax.set_title(
                "[demo-type]-[n-traj]: {0}-{1}".format(
                    self._demo_id_to_demo_quality[selection.params[0].trajectory_num],
                    selection.params[0].pwil_training_param["n_demos"],
                ),
            )
            ax.set_xlabel("Subsampling Frequency")
            ax.set_ylabel("Reward (higher is better)")

    def compare_all_by_demo_id(self):
        stats_optimal = TrajectoriesStats(
            Selector(self._trajectories, self._params)
            .select_by_trajectory_num([0])
            .trajectories
        )
        stats_mixed = TrajectoriesStats(
            Selector(self._trajectories, self._params)
            .select_by_trajectory_num([1, 2, 3])
            .trajectories
        )
        stats_distant = TrajectoriesStats(
            Selector(self._trajectories, self._params)
            .select_by_trajectory_num([4, 5, 6])
            .trajectories
        )

        axes = plt.figure().subplots(1, 3)
        axes[0].plot(stats_optimal.rewards_avg)
        axes[1].plot(stats_mixed.rewards_avg)
        axes[2].plot(stats_distant.rewards_avg)

        plt.show()


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


class SeparateTrajectoriesPlot(TrajectoriesPlot):
    def __init__(self, trajectories: list[np.ndarray]):
        super().__init__(trajectories)

        self._figures = MplUtil(len(self._trajectories)).get_separate_figures()

    def _get_configured_figures(self) -> list[matplotlib.figure.Figure]:
        return self._figures

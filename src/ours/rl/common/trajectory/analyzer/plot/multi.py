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

        self._figure = plt.figure()

    def compare_distant_mixed(self, stats_variant: str):
        figures_upper_lower = self._figure.subfigures(2, 1)

        self.compare_multi_demo_ids(
            figures_upper_lower[0].subplots(1, 3), [1, 2, 3], stats_variant
        )

        self.compare_multi_demo_ids(
            figures_upper_lower[1].subplots(1, 3), [4, 5, 6], stats_variant
        )

        plt.show()

    def compare_distant(self, stats_variant: str = "rewards_avg"):
        self.compare_multi_demo_ids(
            self._figure.subplots(1, 3), [4, 5, 6], stats_variant
        )

        plt.show()

    def compare_mixed(self, stats_variant: str = "rewards_avg"):
        self.compare_multi_demo_ids(
            self._figure.subplots(1, 3), [1, 2, 3], stats_variant
        )

        plt.show()

    def compare_multi_demo_ids(
        self,
        axes: list[mpl.axes.Axes],
        demo_ids: list[int],
        stats_variant: str = "rewards_avg",
    ):
        for ax, demo_id in zip(axes, demo_ids):
            self.compare_one_demo_id(ax, demo_id, True, stats_variant)

    def compare_optimal(
        self, plot_together: bool = True, stats_variant: str = "rewards_avg"
    ):
        self.compare_one_demo_id(
            self._figure.subplots(), 0, plot_together, stats_variant
        )

        plt.show()

    def compare_one_demo_id(
        self,
        ax: mpl.axes.Axes,
        demo_id: int,
        plot_together: bool = True,
        stats_variant: str = "rewards_avg",
    ):
        selections_optimal = []
        for n_demos in [1, 5, 10]:
            selections_optimal.append(
                Selector(self._trajectories, self._params)
                .select_by_trajectory_num([demo_id])
                .select_by_n_demos([n_demos])
            )

        if plot_together:
            self._plot_selections_together(ax, selections_optimal, stats_variant)
        else:
            axes = self._figure.subplots(1, 3)
            self._plot_selections_separate(
                axes, selections_optimal, "subsampling", stats_variant
            )

    def _plot_selections_together(
        self,
        ax: mpl.axes.Axes,
        selections: list[Selector],
        stats_variant: str = "rewards_avg",
    ):
        for selection in selections:
            subsamplings = [
                param.pwil_training_param["subsampling"] for param in selection.params
            ]

            n_demos = selection.params[0].pwil_training_param["n_demos"]
            ax.plot(
                subsamplings,
                self.pick_stats(
                    TrajectoriesStats(selection.trajectories), stats_variant
                ),
                "x--",
                label="num-demos: {0}".format(n_demos),
            )

        ax.set_title(
            "[demo-type]-[n-traj]: {0}-{1}".format(
                self._demo_id_to_demo_quality[selections[0].params[0].trajectory_num],
                "[1 | 5 | 10]",
            ),
        )
        ax.legend()

    def _plot_selections_separate(
        self,
        axes: list[mpl.axes.Axes],
        selections: list[Selector],
        variant: str,
        stats_variant: str = "rewards_avg",
    ):
        for ax, selection in zip(axes, selections):
            self._plot_selection(ax, selection, variant, stats_variant)

    def _plot_selection(
        self,
        ax: mpl.axes.Axes,
        selection: Selector,
        variant: str,
        stats_variant: str = "rewards_avg",
    ):
        if variant == "subsampling":
            subsamplings = [
                param.pwil_training_param["subsampling"] for param in selection.params
            ]

            ax.plot(
                subsamplings,
                self.pick_stats(
                    TrajectoriesStats(selection.trajectories), stats_variant
                ),
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

    def compare_all_by_demo_id(self, variant: str = "rewards_avg"):
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
        axes[0].plot(self.pick_stats(stats_optimal, variant))
        axes[1].plot(self.pick_stats(stats_mixed, variant))
        axes[2].plot(self.pick_stats(stats_distant, variant))

        plt.show()

    @staticmethod
    def pick_stats(stats: TrajectoriesStats, variant: str):
        if variant == "rewards_avg":
            return stats.rewards_avg
        elif variant == "length_avg":
            return stats.lengths_avg
        else:
            exit("wrong stats picker")


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

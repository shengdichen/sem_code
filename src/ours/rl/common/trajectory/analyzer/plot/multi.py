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
    def __init__(
        self,
        trajectories: list[np.ndarray],
        pwil_params: list[PwilParam],
        model_is_discrete: bool = True,
        figure: mpl.figure.FigureBase = None,
    ):
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

        if figure is not None:
            self._figure = figure
        else:
            self._figure = plt.figure()

        self._model_is_discrete = model_is_discrete
        self._env_name = (
            "'PointNav'-Discrete" if model_is_discrete else "'PointNav'-Continuous"
        )

    def plot_mixed_distant(self, stats_variant: str) -> None:
        figures_upper_lower = self._figure.subfigures(2, 1)

        self._plot_multi_demo_ids(
            figures_upper_lower[0].subplots(1, 3), [1, 2, 3], stats_variant
        )

        self._plot_multi_demo_ids(
            figures_upper_lower[1].subplots(1, 3), [4, 5, 6], stats_variant
        )

    def plot_distant(self, stats_variant: str = "rewards_avg") -> None:
        self._plot_multi_demo_ids(self._figure.subplots(1, 3), [4, 5, 6], stats_variant)

    def plot_mixed(self, stats_variant: str = "rewards_avg") -> None:
        self._plot_multi_demo_ids(self._figure.subplots(1, 3), [1, 2, 3], stats_variant)

    def _plot_multi_demo_ids(
        self,
        axes: list[mpl.axes.Axes],
        demo_ids: list[int],
        stats_variant: str = "rewards_avg",
    ) -> None:
        for ax, demo_id in zip(axes, demo_ids):
            selections = self._select_by_demo_id(demo_id)
            self._plot_selections_together(ax, selections, stats_variant)

    def plot_optimal(
        self, plot_together: bool = True, stats_variant: str = "rewards_avg"
    ) -> None:
        if plot_together:
            self._plot_one_demo_id_together(self._figure.subplots(), 0, stats_variant)
        else:
            axes = self._figure.subplots(1, 3)
            self._plot_one_demo_id_separate(axes, 0, stats_variant)

    def _plot_one_demo_id_together(
        self,
        ax: mpl.axes.Axes,
        demo_id: int,
        stats_variant: str = "rewards_avg",
    ) -> None:
        selections = self._select_by_demo_id(demo_id)

        self._plot_selections_together(ax, selections, stats_variant)

    def _plot_one_demo_id_separate(
        self,
        axes: list[mpl.axes.Axes],
        demo_id: int,
        stats_variant: str = "rewards_avg",
    ) -> None:
        selections = self._select_by_demo_id(demo_id)

        self._plot_selections_separate(axes, selections, stats_variant)

    def _select_by_demo_id(self, demo_id: int) -> list[Selector]:
        selections_optimal = []
        for n_demos in [1, 5, 10]:
            selections_optimal.append(
                Selector(self._trajectories, self._params)
                .select_by_trajectory_num([demo_id])
                .select_by_n_demos([n_demos])
            )

        return selections_optimal

    def _plot_selections_together(
        self,
        ax: mpl.axes.Axes,
        selections: list[Selector],
        stats_variant: str = "rewards_avg",
    ) -> None:
        marker_styles = ["x", "+", "."]
        markersize_styles = [6.0, 7.5, 9.5]
        for selection, marker_style, markersize_style in zip(
            selections, marker_styles, markersize_styles
        ):
            subsamplings = [
                param.pwil_training_param["subsampling"] for param in selection.params
            ]

            n_demos = selection.params[0].pwil_training_param["n_demos"]
            ax.plot(
                subsamplings,
                self._pick_stats(
                    TrajectoriesStats(selection.trajectories), stats_variant
                ),
                marker=marker_style,
                markersize=markersize_style,
                dashes=[5, 3],
                label="num-demos: {0}".format(n_demos),
            )

        ax.set_title(
            "{0}\n<demo-id={1}({2})> -- <n-demos={3}>".format(
                self._env_name,
                selections[0].params[0].trajectory_num,
                self._demo_id_to_demo_quality[selections[0].params[0].trajectory_num],
                "[1 | 5 | 10]",
            ),
        )
        self._set_axis_labels(ax, stats_variant)
        ax.legend()

    def _plot_selections_separate(
        self,
        axes: list[mpl.axes.Axes],
        selections: list[Selector],
        stats_variant: str = "rewards_avg",
    ) -> None:
        for ax, selection in zip(axes, selections):
            self._plot_selection(ax, selection, stats_variant)

    def _plot_selection(
        self,
        ax: mpl.axes.Axes,
        selection: Selector,
        stats_variant: str = "rewards_avg",
    ) -> None:
        subsamplings = [
            param.pwil_training_param["subsampling"] for param in selection.params
        ]

        ax.plot(
            subsamplings,
            self._pick_stats(TrajectoriesStats(selection.trajectories), stats_variant),
            "x--",
        )
        ax.set_title(
            "[demo-type]-[n-traj]: {0}({1})-{2}".format(
                selection.params[0].trajectory_num,
                self._demo_id_to_demo_quality[selection.params[0].trajectory_num],
                selection.params[0].pwil_training_param["n_demos"],
            ),
        )
        self._set_axis_labels(ax, stats_variant)

    def _set_axis_labels(
        self, ax: mpl.axes.Axes, stats_variant: str = "rewards_avg"
    ) -> None:
        ax.set_xlabel("Subsampling Frequency")
        ax.set_ylim([0, 1200])

        baseline_line_style = {
            "color": "rebeccapurple",
            "label": "baseline",
            "linewidth": 2.5,
        }
        expert_line_style = {"color": "grey", "label": "expert", "linewidth": 2.5}
        ax.axhline(
            950,  # baseline
            **baseline_line_style,
        )
        if self._model_is_discrete:
            ax.axhline(208, **expert_line_style)
        else:
            ax.axhline(88.2, **expert_line_style)

        if stats_variant == "rewards_avg":
            ax.set_ylabel("Reward (higher is better)")
        else:
            ax.set_ylabel("Length (lower is better)")

    def plot_all_types(self, variant: str = "rewards_avg") -> None:
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
        axes[0].plot(self._pick_stats(stats_optimal, variant))
        axes[1].plot(self._pick_stats(stats_mixed, variant))
        axes[2].plot(self._pick_stats(stats_distant, variant))

    @staticmethod
    def _pick_stats(stats: TrajectoriesStats, variant: str) -> np.ndarray:
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

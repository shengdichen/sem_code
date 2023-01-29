import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from src.ours.eval.pointenv.illustration.trajectory.select import Selector
from src.ours.rl.common.trajectory.analyzer.stats.multi import TrajectoriesStats
from src.ours.rl.pwil.param import PwilParam


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
        ax.set_xticks([1, 2, 5, 10, 20])
        if stats_variant == "length_avg":
            ax.set_ylim([0, 1200])
        else:
            ax.set_ylim([-1.35e5, 0])

        baseline_line_style = {
            "color": "rebeccapurple",
            "label": "baseline",
            "linewidth": 2.5,
        }
        expert_line_style = {"color": "grey", "label": "expert", "linewidth": 2.5}
        if stats_variant == "length_avg":
            ax.axhline(
                950,  # baseline
                **baseline_line_style,
            )
            if self._model_is_discrete:
                ax.axhline(208, **expert_line_style)
            else:
                ax.axhline(88.2, **expert_line_style)
        else:
            ax.axhline(
                -7e4,  # baseline
                **baseline_line_style,
            )
            if self._model_is_discrete:
                ax.axhline(-1.3e4, **expert_line_style)
            else:
                ax.axhline(-5300, **expert_line_style)

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

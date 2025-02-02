import matplotlib
import numpy as np

from src.ours.rl.common.trajectory.analyzer.info import TrajectoryInfo


class TrajectoryPlotAtom:
    def __init__(self, trajectory: np.ndarray, ax: matplotlib.axes.Axes):
        self._info = TrajectoryInfo(trajectory)

        self._ax = ax
        self._canvas_size = 200
        self._bins_hist = self._make_bins_hist()

    def _make_bins_hist(self, nr=40) -> tuple[np.ndarray, np.ndarray]:
        x_bins = np.linspace(0, self._canvas_size, nr)
        y_bins = np.linspace(0, self._canvas_size, nr)

        return x_bins, y_bins

    def plot_agent(self, plot_hist: bool) -> None:
        if plot_hist:
            self._plot_agent_hist()
        else:
            self._plot_agent_direct()

    def _plot_agent_hist(self) -> None:
        self._ax.set_title("Agent Position [histogram]")
        self._make_square()
        agent_pos_x, agent_pos_y = self._info.agent_pos
        self._ax.hist2d(agent_pos_x, agent_pos_y, bins=self._bins_hist)

    def _plot_agent_direct(self) -> None:
        self._ax.set_title("Agent Position [direct plot]")
        self._make_square()
        agent_pos_x, agent_pos_y = self._info.agent_pos
        self._ax.plot(agent_pos_x, agent_pos_y, "m-", alpha=0.3)

    def plot_target(self) -> None:
        self._ax.set_title("Target Position")
        self._make_square()
        target_pos_x, target_pos_y = self._info.target_pos
        self._ax.scatter(target_pos_x, target_pos_y, c="r")

    def plot_action(self) -> None:
        self._ax.set_title("Model's Action")

        action = self._info.action
        if action.shape[1] == 1:
            action_space_size = 5
            self._ax.hist(
                action,
                bins=(
                    np.arange(action_space_size + 1) - 0.5
                ),  # align&center hist-bins with the action-values
                rwidth=0.6,  # add space between hist-bars
            )
            self._ax.set_xticks(
                np.arange(action_space_size)  # show only integer-valued ticks
            )
        elif action.shape[1] == 2:
            labels = ["dimension-1", "dimension-2"]
            self._ax.hist(action, bins=20, label=labels)
            self._ax.legend()

    def _make_square(self) -> None:
        self._ax.set_xlim(0, self._canvas_size)
        self._ax.set_ylim(0, self._canvas_size)
        self._ax.set_aspect(1)
        self._ax.set_box_aspect(1)


class TrajectoryPlot:
    def __init__(
        self,
        trajectory: np.ndarray,
        figure: matplotlib.figure.Figure | matplotlib.figure.SubFigure,
    ):
        self._trajectory = trajectory
        self._figure = figure

    def plot_agent_and_target(self, plot_agent_as_hist: bool) -> None:
        axs = self._figure.subplots(1, 2)

        TrajectoryPlotAtom(self._trajectory, axs[0]).plot_agent(plot_agent_as_hist)
        TrajectoryPlotAtom(self._trajectory, axs[1]).plot_target()

    def plot_agent_and_action(self, plot_agent_as_hist: bool) -> None:
        axs = self._figure.subplots(1, 2)

        TrajectoryPlotAtom(self._trajectory, axs[0]).plot_agent(plot_agent_as_hist)
        TrajectoryPlotAtom(self._trajectory, axs[1]).plot_action()

    def plot_agent_target_action(self) -> None:
        subfigure_upper, subfigure_lower = self._figure.subfigures(2, 1)

        axes_upper = subfigure_upper.subplots(1, 2)
        TrajectoryPlotAtom(self._trajectory, axes_upper[0]).plot_target()
        TrajectoryPlotAtom(self._trajectory, axes_upper[0]).plot_agent(False)
        TrajectoryPlotAtom(self._trajectory, axes_upper[1]).plot_target()

        axes_lower = subfigure_lower.subplots(1, 2)
        TrajectoryPlotAtom(self._trajectory, axes_lower[0]).plot_agent(True)
        TrajectoryPlotAtom(self._trajectory, axes_lower[1]).plot_action()

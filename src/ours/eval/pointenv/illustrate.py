import matplotlib as mpl
from matplotlib import pyplot as plt

from src.ours.eval.pointenv.illustration.trajectory.compare import (
    TrajectoriesComparisonPlot,
)
from src.ours.eval.pointenv.pwil import (
    DiscretePointNavPwilManager,
    ContPointNavPwilManager,
    PointNavPwilParams,
)


class TrajectoriesAnalysisPlotConfig:
    use_length_as_metric: bool = True


class TrajectoriesAnalysisPlot:
    def __init__(self):
        self._figure = plt.figure()
        plt.rcParams["font.family"] = "Fira Code"
        plt.rcParams["font.size"] = 14.5

        self._trajectories_discrete, self._trajectories_cont = (
            [
                manager.load_trajectory()
                for manager in DiscretePointNavPwilManager().managers
            ],
            [
                manager.load_trajectory()
                for manager in ContPointNavPwilManager().managers
            ],
        )
        self._params = PointNavPwilParams().get_params()

        self._config = TrajectoriesAnalysisPlotConfig

    def _make_discrete_plotter(self, figure: mpl.figure.FigureBase):
        return TrajectoriesComparisonPlot(
            self._trajectories_discrete,
            self._params,
            model_is_discrete=True,
            figure=figure,
        )

    def _make_cont_plotter(self, figure: mpl.figure.FigureBase):
        return TrajectoriesComparisonPlot(
            self._trajectories_cont,
            self._params,
            model_is_discrete=False,
            figure=figure,
        )

    def plot_optimals(self):
        figures = self._figure.subfigures(2, 2)

        self._make_discrete_plotter(figures[0][0]).plot_optimal(
            plot_together=True, stats_variant="length_avg"
        )
        self._make_cont_plotter(figures[0][1]).plot_optimal(
            plot_together=True, stats_variant="length_avg"
        )

        self._make_discrete_plotter(figures[1][0]).plot_optimal(
            plot_together=True, stats_variant="rewards_avg"
        )
        self._make_cont_plotter(figures[1][1]).plot_optimal(
            plot_together=True, stats_variant="rewards_avg"
        )

        self._figure.set_size_inches(20, 20)
        self._figure.set_dpi(200)
        self._figure.savefig("optimal.png")

    def plot_mixed_distant_discrete(self):
        if self._config.use_length_as_metric:
            self._make_discrete_plotter(self._figure).plot_mixed_distant(
                stats_variant="length_avg"
            )
        else:
            self._make_discrete_plotter(self._figure).plot_mixed_distant(
                stats_variant="rewards_avg"
            )

        self._figure.set_size_inches(27, 20)
        self._figure.set_dpi(200)
        self._figure.savefig("mixed_distant_discrete.png")

    def plot_mixed_distant_cont(self):
        if self._config.use_length_as_metric:
            self._make_cont_plotter(self._figure).plot_mixed_distant(
                stats_variant="length_avg"
            )
        else:
            self._make_cont_plotter(self._figure).plot_mixed_distant(
                stats_variant="rewards_avg"
            )

        self._figure.set_size_inches(27, 20)
        self._figure.set_dpi(200)
        self._figure.savefig("mixed_distant_cont.png")


def client_code():
    trainer = TrajectoriesAnalysisPlot()
    trainer.plot_mixed_distant_cont()
    plt.show()


if __name__ == "__main__":
    client_code()

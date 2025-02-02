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
        self._figure.set_dpi(300)
        plt.rcParams["font.family"] = "Fira Code"
        plt.rcParams["font.size"] = 14.5
        self._savedir = "./illustration/"

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

    def _make_discrete_plotter(
        self, figure: mpl.figure.FigureBase
    ) -> TrajectoriesComparisonPlot:
        return TrajectoriesComparisonPlot(
            self._trajectories_discrete,
            self._params,
            model_is_discrete=True,
            figure=figure,
        )

    def _make_cont_plotter(
        self, figure: mpl.figure.FigureBase
    ) -> TrajectoriesComparisonPlot:
        return TrajectoriesComparisonPlot(
            self._trajectories_cont,
            self._params,
            model_is_discrete=False,
            figure=figure,
        )

    def plot_optimals(self) -> None:
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
        self._figure.savefig(self._savedir + "optimal.png")

    def plot_mixed_distant_discrete(self) -> None:
        if self._config.use_length_as_metric:
            self._make_discrete_plotter(self._figure).plot_mixed_distant(
                stats_variant="length_avg"
            )
        else:
            self._make_discrete_plotter(self._figure).plot_mixed_distant(
                stats_variant="rewards_avg"
            )

        self._set_figure_mixed_distant()

        self._save_figure_mixed_distant("mixed_distant_discrete")

    def plot_mixed_distant_cont(self) -> None:
        if self._config.use_length_as_metric:
            self._make_cont_plotter(self._figure).plot_mixed_distant(
                stats_variant="length_avg"
            )
        else:
            self._make_cont_plotter(self._figure).plot_mixed_distant(
                stats_variant="rewards_avg"
            )

        self._set_figure_mixed_distant()

        self._save_figure_mixed_distant("mixed_distant_cont")

    def _set_figure_mixed_distant(self) -> None:
        self._figure.set_size_inches(20, 30)

    def _save_figure_mixed_distant(self, filename_base: str) -> None:
        if self._config.use_length_as_metric:
            plot_variant = "_length"
        else:
            plot_variant = "_reward"

        self._figure.savefig(self._savedir + filename_base + plot_variant + ".png")


def client_code():
    trainer = TrajectoriesAnalysisPlot()
    trainer.plot_mixed_distant_cont()
    plt.show()


if __name__ == "__main__":
    client_code()

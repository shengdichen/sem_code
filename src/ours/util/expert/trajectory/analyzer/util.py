import matplotlib
from matplotlib import pyplot as plt


class MplUtil:
    def __init__(self, n_trajectories: int):
        self._n_trajectories = n_trajectories

    def get_horizontal_figures(
        self,
        figure: matplotlib.figure.Figure | matplotlib.figure.SubFigure = None,
    ) -> list[matplotlib.figure.SubFigure]:
        if figure is None:
            figure = plt.figure(figsize=[15, 5])
        if self._n_trajectories == 1:
            return [figure.subfigures(1, 1)]
        else:
            return figure.subfigures(1, self._n_trajectories)

    def get_separate_figures(self) -> list[matplotlib.figure.Figure]:
        return [plt.figure(figsize=[15, 5]) for __ in range(self._n_trajectories)]

    @staticmethod
    def show_figures() -> None:
        plt.show()

import matplotlib
from matplotlib import pyplot as plt


class MplUtil:
    def __init__(self, n_trajectories: int):
        self._n_trajectories = n_trajectories

    def get_parallel_figures(
        self,
        figure: matplotlib.figure.Figure
        | matplotlib.figure.SubFigure = plt.figure(figsize=[15, 5]),
    ) -> list[matplotlib.figure.Figure] | list[matplotlib.figure.SubFigure]:
        if self._n_trajectories == 1:
            return [figure]
        else:
            return figure.subfigures(1, self._n_trajectories)

    def get_separate_figures(self) -> list[matplotlib.figure.Figure]:
        return [plt.figure(figsize=[15, 5]) for __ in range(self._n_trajectories)]

    @staticmethod
    def show_figures() -> None:
        plt.show()

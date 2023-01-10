import numpy as np
from PIL import Image
from gym import Env
from matplotlib import pyplot as plt

from src.ours.util.common.saveload import NumpySaveLoad
from src.ours.util.pwil.rewardplot.rewardplotter import RewardPlotter
from src.ours.util.common.param import PwilParam
from src.ours.util.common.pathprovider import PwilSaveLoadPathGenerator


class RewardPlotManager:
    def __init__(
        self,
        training_param: PwilParam,
        env_pwil_rewarded_and_identifier: tuple[Env, str],
    ):
        self._env_pwil_rewarded, env_identifier = env_pwil_rewarded_and_identifier

        self._reward_plot = self._make_reward_plot()
        self._path_saveload = PwilSaveLoadPathGenerator(training_param).get_plot_path(
            env_identifier
        )
        self._saveloader_numpy = NumpySaveLoad(self._path_saveload)

    @property
    def reward_plot(self) -> np.ndarray:
        return self._reward_plot

    def _make_reward_plot(self) -> np.ndarray:
        plot = RewardPlotter.plot_reward(
            discriminator=None, env=self._env_pwil_rewarded
        )

        return plot

    def save_reward_plot(self, save_np: bool = True) -> None:
        im = Image.fromarray(self._reward_plot)

        save_path = str(self._path_saveload) + ".png"
        im.save(save_path)

        if save_np:
            self.save_reward_plot_np()

    def show_reward_plot(self) -> None:
        ax = plt.figure().subplots()
        ax.imshow(self._reward_plot)
        plt.show()

    def save_reward_plot_np(self) -> None:
        self._saveloader_numpy.save(self._reward_plot)

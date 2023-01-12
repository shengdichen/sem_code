import numpy as np
from PIL import Image
from gym import Env
from matplotlib import pyplot as plt

from src.ours.util.common.param import PwilParam
from src.ours.util.common.pathprovider import PwilSaveLoadPathGenerator
from src.ours.util.common.saveload import NumpySaveLoad
from src.ours.util.pwil.rewardplot.rewardplotter import RewardPlotter


class RewardPlotConfig:
    def __init__(self):
        self._force_regenerate: bool = False

        self._save_as_image: bool = True
        self._save_as_numpy: bool = True

    @property
    def force_regenerate(self) -> bool:
        return self._force_regenerate

    @force_regenerate.setter
    def force_regenerate(self, value: bool) -> None:
        self._force_regenerate = value

    @property
    def save_as_image(self) -> bool:
        return self._save_as_image

    @save_as_image.setter
    def save_as_image(self, value: bool) -> None:
        self._save_as_image = value

    @property
    def save_as_numpy(self) -> bool:
        return self._save_as_numpy

    @save_as_numpy.setter
    def save_as_numpy(self, value: bool) -> None:
        self._save_as_numpy = value


class RewardPlotManager:
    def __init__(
        self,
        training_param: PwilParam,
        env_pwil_rewarded_and_identifier: tuple[Env, str],
        reward_plot_config: RewardPlotConfig = RewardPlotConfig(),
    ):
        self._config = reward_plot_config

        self._env_pwil_rewarded, env_identifier = env_pwil_rewarded_and_identifier

        self._path_saveload = PwilSaveLoadPathGenerator(training_param).get_plot_path(
            env_identifier
        )
        self._saveloader_numpy = NumpySaveLoad(self._path_saveload)

        self._reward_plot = self._make_reward_plot()

    @property
    def reward_plot(self) -> np.ndarray:
        return self._reward_plot

    def _make_reward_plot(self) -> np.ndarray:
        if self._config.force_regenerate:
            reward_plot = self._get_reward_plot()
        elif self._saveloader_numpy.exists():
            reward_plot = self._saveloader_numpy.load()
        else:
            reward_plot = self._get_reward_plot()

        return reward_plot

    def _get_reward_plot(self) -> np.ndarray:
        plot = RewardPlotter.plot_reward(
            discriminator=None, env=self._env_pwil_rewarded
        )

        return plot

    def save(self) -> None:
        if self._config.save_as_image:
            self._save_as_image()
        if self._config.save_as_numpy:
            self._saveloader_numpy.save(self._reward_plot)

    def _save_as_image(self) -> None:
        im = Image.fromarray(self._reward_plot)

        save_path = str(self._path_saveload) + ".png"
        im.save(save_path)

    def show_reward_plot(self) -> None:
        ax = plt.figure().subplots()
        ax.imshow(self._reward_plot)
        plt.show()

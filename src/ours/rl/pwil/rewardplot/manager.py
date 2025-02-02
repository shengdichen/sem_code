import numpy as np
from gym import Env
from matplotlib import pyplot as plt

from src.ours.rl.common.saveload.image import ImageSaveLoad
from src.ours.rl.common.saveload.numpy import NumpySaveLoad
from src.ours.rl.pwil.param import PwilParam
from src.ours.rl.pwil.path import PwilSaveLoadPathGenerator
from src.ours.rl.pwil.rewardplot.rewardplotter import RewardPlotter


class RewardPlotConfig:
    def __init__(self):
        self._auto_load: bool = True
        self._force_regenerate: bool = False

        self._save_as_image: bool = True
        self._save_as_numpy: bool = True

    @property
    def auto_load(self):
        return self._auto_load

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

        self._path_saveload = PwilSaveLoadPathGenerator(
            env_identifier, training_param
        ).get_rewardplot_path()
        self._saveloader_numpy = NumpySaveLoad(self._path_saveload)
        self._saveloader_image = ImageSaveLoad(self._path_saveload)

        if self._config.auto_load:
            self._reward_plot = self._make_reward_plot()
        else:
            self._reward_plot = None

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
        if self._reward_plot is None:
            self._reward_plot = self._make_reward_plot()

        if self._config.save_as_image:
            self._saveloader_image.save_from_np(self._reward_plot, force_resave=False)
        if self._config.save_as_numpy:
            self._saveloader_numpy.save(self._reward_plot, force_resave=False)

    def show_reward_plot(self) -> None:
        if self._reward_plot is None:
            self._reward_plot = self._make_reward_plot()

        ax = plt.figure().subplots()
        ax.imshow(self._reward_plot)
        plt.show()

import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from stable_baselines3 import PPO

from src.ours.env.creation import (
    PointEnvFactory,
    PointEnvIdentifierGenerator,
    PointEnvConfigFactory,
)
from src.ours.env.env import MovePoint
from src.ours.eval.pointenv.expert import PointEnvExpertDefault
from src.ours.util.common.param import PwilParam
from src.ours.util.common.helper import RewardPlotter
from src.ours.util.pwil.train import (
    PwilManagerFactory,
    PwilManager,
)
from src.upstream.env_utils import PWILReward


class PointEnvPwilManagerFactory:
    def __init__(self, training_param: PwilParam = PwilParam()):
        self._training_param = training_param

        env_config = PointEnvConfigFactory().env_configs[0]
        self._env_raw, self._env_raw_testing = (
            PointEnvFactory(env_config).create(),
            PointEnvFactory(env_config).create(),
        )
        self._env_identifier = PointEnvIdentifierGenerator().from_env(self._env_raw)

        self._demos_all = self._get_all_demos()
        self._demos_selected = self._demos_all[training_param.trajectory_num]

    @property
    def pwil_manager(self) -> PwilManager:
        return PwilManagerFactory(
            self._training_param,
            ((self._env_raw, self._env_raw_testing), self._env_identifier),
            self._demos_selected,
        ).pwil_manager

    @staticmethod
    def _get_all_demos():
        pointenv_expert_default = PointEnvExpertDefault()

        demos = pointenv_expert_default._load()
        flat_demos = [item for sublist in demos for item in sublist]
        flat_demos_0 = [item for sublist in demos for item in sublist]
        flat_demos_01 = [item for sublist in demos[:1] for item in sublist]
        flat_demos_12 = [item for sublist in demos[1:] for item in sublist]

        return flat_demos, flat_demos_0, flat_demos_01, flat_demos_12

    def set_pwil_training_param(
        self, n_demos: int = None, subsampling: int = None, use_actions: bool = False
    ) -> "PointEnvPwilManagerFactory":
        pwil_training_param = self._training_param.pwil_training_param

        if n_demos is not None:
            pwil_training_param["n_demos"] = n_demos

        if subsampling is not None:
            pwil_training_param["subsampling"] = subsampling

        if use_actions is not None:
            pwil_training_param["use_actions"] = use_actions

        return self

    def set_trajectories(
        self, trajectories_num: int = None
    ) -> "PointEnvPwilManagerFactory":
        if trajectories_num is not None:
            self._demos_selected = self._demos_all[trajectories_num]
            self._training_param.trajectory_num = trajectories_num

        return self


class ClientTrainerPwil:
    def __init__(self):
        self._managers = []
        for demo_id in [0, 1, 2, 3]:
            for n_demos in [1, 2, 3]:
                for subsampling in [1, 2, 3, 5, 10, 20]:
                    print(
                        "subsampling: ",
                        subsampling,
                        " dem: ",
                        demo_id,
                        " n_demos: ",
                        n_demos,
                    )
                    manager_factory = (
                        PointEnvPwilManagerFactory()
                        .set_pwil_training_param(
                            n_demos=n_demos, subsampling=subsampling
                        )
                        .set_trajectories(demo_id)
                    )
                    self._managers.append(manager_factory.pwil_manager)

    def train_and_save(self) -> None:
        for manager in self._managers:
            manager.train_and_save()

    def save_plot_with_torch(self) -> None:
        plots = []

        for manager in self._managers:
            plots.append(manager.get_reward_plot())

        torchvision.utils.save_image(plots, normalize=True, nrow=6)

    def test(self) -> None:
        for manager in self._managers:
            manager.test_model()


def client_code():
    trainer = ClientTrainerPwil()
    trainer.test()


if __name__ == "__main__":
    client_code()

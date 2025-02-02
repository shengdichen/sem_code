import logging

import numpy as np
import torchvision

from src.ours.env.config import PointNavConfigFactory
from src.ours.env.creation import (
    DiscretePointNavFactory,
    ContPointNavFactory,
)
from src.ours.env.env import PointNav, DiscretePointNav, ContPointNav
from src.ours.env.identifier import (
    DiscretePointNavIdentifierGenerator,
    ContPointNavIdentifierGenerator,
)
from src.ours.eval.common.action_provider import ActionProvider
from src.ours.eval.pointenv.expert import (
    DiscretePointNavExpertDefault,
    ContPointNavExpertDefault,
)
from src.ours.eval.pointenv.run.run import DiscretePointNavRunner, ContPointNavRunner
from src.ours.rl.pwil.manager import (
    PwilManagerFactory,
    PwilManager,
)
from src.ours.rl.pwil.param import PwilParam

logger = logging.getLogger(__name__)


class PointNavPwilManagerFactory:
    def __init__(self, trajectories: list[np.ndarray]):
        self._trajectories = trajectories

        demonstration_0 = self._convert_selected_trajectories([0])

        (demonstration_01, demonstration_02, demonstration_012) = (
            self._convert_selected_trajectories([0, 1]),
            self._convert_selected_trajectories([0, 2]),
            self._convert_selected_trajectories([0, 1, 2]),
        )

        demonstration_1, demonstration_2, demonstration_12 = (
            self._convert_selected_trajectories([1]),
            self._convert_selected_trajectories([2]),
            self._convert_selected_trajectories([1, 2]),
        )

        self._demonstrations = (
            demonstration_0,
            demonstration_01,
            demonstration_02,
            demonstration_012,
            demonstration_1,
            demonstration_2,
            demonstration_12,
        )

    def _convert_selected_trajectories(
        self,
        selected_indexes: list[int],
    ) -> list[np.ndarray]:
        demonstration = []
        for index in selected_indexes:
            demonstration.extend(self._trajectories[index])

        return demonstration

    def get_pwil_managers(self) -> list[PwilManager]:
        managers = []
        for pwil_param in PointNavPwilParams().get_params():
            logging.info(pwil_param.get_pwil_hyperparam_info())
            managers.append(self._get_pwil_manager(pwil_param))

        return managers

    def _get_pwil_manager(self, training_param: PwilParam) -> PwilManager:
        (env_raw, env_eval), env_identifier = self._get_envs_and_identifier()

        return PwilManagerFactory(
            training_param,
            ((env_raw, env_eval), env_identifier),
            self._demonstrations[training_param.trajectory_num],
        ).pwil_manager

    def _get_envs_and_identifier(
        self,
    ) -> tuple[tuple[PointNav, PointNav], str]:
        pass


class DiscretePointNavPwilManagerFactory(PointNavPwilManagerFactory):
    def __init__(self):
        super().__init__(DiscretePointNavExpertDefault().load_trajectories())

    def _get_envs_and_identifier(
        self,
    ) -> tuple[tuple[DiscretePointNav, DiscretePointNav], str]:
        env_config = PointNavConfigFactory().env_configs[0]
        env_raw, env_eval = (
            DiscretePointNavFactory(env_config).create(),
            DiscretePointNavFactory(env_config).create(),
        )
        env_identifier = DiscretePointNavIdentifierGenerator().from_env(env_raw)

        return (env_raw, env_eval), env_identifier


class ContPointNavPwilManagerFactory(PointNavPwilManagerFactory):
    def __init__(self):
        super().__init__(ContPointNavExpertDefault().load_trajectories())

    def _get_envs_and_identifier(
        self,
    ) -> tuple[tuple[ContPointNav, ContPointNav], str]:
        env_config = PointNavConfigFactory().env_configs[0]
        env_raw, env_eval = (
            ContPointNavFactory(env_config).create(),
            ContPointNavFactory(env_config).create(),
        )
        env_identifier = ContPointNavIdentifierGenerator().from_env(env_raw)

        return (env_raw, env_eval), env_identifier


class PointNavPwilParams:
    def __init__(self):
        self._n_demos_pool = [1, 5, 10]
        self._subsampling_pool = [1, 2, 5, 10, 20]

    def get_params(self) -> list[PwilParam]:
        return self._get_params([0, 1, 2, 3, 4, 5, 6])

    def get_optimal_params(self) -> list[PwilParam]:
        return self._get_params([0])

    def get_mixed_params(self) -> list[PwilParam]:
        return self._get_params([1, 2, 3])

    def get_distant_params(self) -> list[PwilParam]:
        return self._get_params([4, 5, 6])

    def get_best_params(self) -> list[PwilParam]:
        demo_id = 0
        n_demos = 1
        subsampling_pool = 1

        return [self._convert_to_param((demo_id, n_demos, subsampling_pool))]

    def _get_params(self, demo_id_pool: list[int]) -> list[PwilParam]:
        configs = []
        for demo_id in demo_id_pool:
            for n_demos in self._n_demos_pool:
                for subsampling in self._subsampling_pool:
                    configs.append(
                        self._convert_to_param((demo_id, n_demos, subsampling))
                    )
        return configs

    @staticmethod
    def _convert_to_param(config: tuple[int, int, int]) -> PwilParam:
        demo_id, n_demos, subsampling = config

        pwil_param = PwilParam()
        pwil_param.trajectory_num = demo_id
        pwil_param.pwil_training_param["n_demos"] = n_demos
        pwil_param.pwil_training_param["subsampling"] = subsampling

        return pwil_param


class PointNavPwilManager:
    def __init__(self, managers: list[PwilManager]):
        self._managers = managers

    @property
    def managers(self) -> list[PwilManager]:
        return self._managers

    def save_rewardplots(self) -> None:
        for manager in self._managers:
            manager.save_reward_plot()

    def save_rewardplots_with_torch(self) -> None:
        plots = []

        for manager in self._managers:
            plots.append(manager.get_reward_plot())

        torchvision.utils.save_image(plots, normalize=True, nrow=6)

    def train_and_save_models(self) -> None:
        for manager in self._managers:
            manager.train_model()
            manager.save_model()

    def save_trajectories_and_stats_and_plot(self):
        self.save_trajectories()
        self.save_trajectories_stats_and_plot()

    def save_trajectories(self):
        for manager in self._managers:
            manager.save_trajectory()

    def save_trajectories_stats_and_plot(self):
        for manager in self._managers:
            manager.save_trajectory_stats_and_plot()

    def test_models(self) -> None:
        for manager in self._managers:
            manager.test_model()

    def run_models(self) -> None:
        pass


class DiscretePointNavPwilManager(PointNavPwilManager):
    def __init__(self):
        super().__init__(DiscretePointNavPwilManagerFactory().get_pwil_managers())

    def run_models(self) -> None:
        model = self._managers[0].load_model()

        class ActionProviderModel(ActionProvider):
            def get_action(self, obs: np.ndarray, **kwargs):
                return model.predict(obs)[0]

        DiscretePointNavRunner().run_episodes(ActionProviderModel())


class ContPointNavPwilManager(PointNavPwilManager):
    def __init__(self):
        super().__init__(ContPointNavPwilManagerFactory().get_pwil_managers())

    def run_models(self) -> None:
        model = self._managers[0].load_model()

        class ActionProviderModel(ActionProvider):
            def get_action(self, obs: np.ndarray, **kwargs):
                return model.predict(obs)[0]

        ContPointNavRunner().run_episodes(ActionProviderModel())


def client_code():
    trainer = DiscretePointNavPwilManager()
    trainer.test_models()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(module)s: [%(levelname)s] %(message)s", level=logging.INFO
    )

    client_code()

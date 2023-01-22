import numpy as np
import torchvision

from src.ours.env.creation import (
    PointEnvFactory,
    PointEnvIdentifierGenerator,
    PointEnvConfigFactory,
)
from src.ours.eval.pointenv.expert import PointEnvExpertDefault
from src.ours.eval.pointenv.run.run import PointEnvRunner
from src.ours.eval.pointenv.run.actionprovider import ActionProvider
from src.ours.util.common.param import PwilParam
from src.ours.util.pwil.manager import (
    PwilManagerFactory,
    PwilManager,
)


class PointEnvPwilManagerFactory:
    def __init__(self, training_param: PwilParam = PwilParam()):
        self._training_param = training_param

        env_config = PointEnvConfigFactory().env_configs[0]
        self._env_raw, self._env_eval = (
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
            ((self._env_raw, self._env_eval), self._env_identifier),
            self._demos_selected,
        ).pwil_manager

    @staticmethod
    def _get_all_demos():
        pointenv_expert_default = PointEnvExpertDefault()

        trajectories = pointenv_expert_default.load_trajectories()

        trajectory_0 = PointEnvPwilManagerFactory.make_selected_trajectories(
            trajectories, [0]
        )

        (trajectory_01, trajectory_02, trajectory_012) = (
            PointEnvPwilManagerFactory.make_selected_trajectories(trajectories, [0, 1]),
            PointEnvPwilManagerFactory.make_selected_trajectories(trajectories, [0, 2]),
            PointEnvPwilManagerFactory.make_all_tractories(trajectories),
        )

        trajectory_1, trajectory_2, trajectory_12 = (
            PointEnvPwilManagerFactory.make_selected_trajectories(trajectories, [1]),
            PointEnvPwilManagerFactory.make_selected_trajectories(trajectories, [2]),
            PointEnvPwilManagerFactory.make_selected_trajectories(trajectories, [1, 2]),
        )

        return (
            trajectory_0,
            trajectory_01,
            trajectory_02,
            trajectory_012,
            trajectory_1,
            trajectory_2,
            trajectory_12,
        )

    @staticmethod
    def make_selected_trajectories(
        trajectories: list[np.ndarray],
        selected_indexes: list[int],
    ) -> list[np.ndarray]:
        selected_trajectories = []
        for index in selected_indexes:
            selected_trajectories.extend(trajectories[index])

        return selected_trajectories

    @staticmethod
    def make_all_tractories(trajectories: list[np.ndarray]) -> list[np.ndarray]:
        selected_trajectories = []
        for trajectory in trajectories:
            selected_trajectories.extend(trajectory)

        return selected_trajectories

    def set_pwil_training_param(
        self, n_demos: int = None, subsampling: int = None, use_actions: bool = None
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


class PointEnvTrajectoryPool:
    def __init__(self):
        self._trajectories = PointEnvExpertDefault().load_trajectories()

        demonstration_0 = self._convert_selected_trajectories([0])

        (demonstration_01, demonstration_02, demonstration_012) = (
            self._convert_selected_trajectories([0, 1]),
            self._convert_selected_trajectories([0, 2]),
            self._convert_all_tractories(),
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

    def _convert_all_tractories(self) -> list[np.ndarray]:
        demonstration = []
        for trajectory in self._trajectories:
            demonstration.extend(trajectory)

        return demonstration

    def get_demonstration(self, demonstration_id: int) -> list[np.ndarray]:
        return self._demonstrations[demonstration_id]


class PointEnvPwilConfig:
    def __init__(self):
        self._n_demos_pool = [1, 5, 10]
        self._subsampling_pool = [1, 2, 5, 10, 20]

    def _get_configs(self, demo_id_pool: list[int]) -> list[tuple[int, int, int]]:
        configs = []
        for demo_id in demo_id_pool:
            for n_demos in self._n_demos_pool:
                for subsampling in self._subsampling_pool:
                    configs.append((demo_id, n_demos, subsampling))
        return configs

    def get_configs(self) -> list[tuple[int, int, int]]:
        return self._get_configs([0, 1, 2, 3, 4, 5, 6])

    def get_optimal_configs(self) -> list[tuple[int, int, int]]:
        return self._get_configs([0])

    def get_mixed_configs(self) -> list[tuple[int, int, int]]:
        return self._get_configs([1, 2, 3])

    def get_distant_configs(self) -> list[tuple[int, int, int]]:
        return self._get_configs([4, 5, 6])

    @staticmethod
    def get_best_config() -> list[tuple[int, int, int]]:
        demo_id = 0
        n_demos = 1
        subsampling_pool = 1

        return [(demo_id, n_demos, subsampling_pool)]


class PointEnvPwilManager:
    def __init__(self):
        self._managers = []
        for demo_id, n_demos, subsampling in PointEnvPwilConfig().get_configs():
            print(
                " dem: ",
                demo_id,
                " n_demos: ",
                n_demos,
                "subsampling: ",
                subsampling,
            )
            manager_factory = (
                PointEnvPwilManagerFactory()
                .set_pwil_training_param(n_demos=n_demos, subsampling=subsampling)
                .set_trajectories(demo_id)
            )
            self._managers.append(manager_factory.pwil_manager)

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
        model = self._managers[0].load_model()

        class ActionProviderModel(ActionProvider):
            def get_action(self, obs: np.ndarray, **kwargs):
                return model.predict(obs)[0]

        PointEnvRunner().run_episodes(ActionProviderModel())


def client_code():
    trainer = PointEnvPwilManager()
    trainer.test_models()


if __name__ == "__main__":
    client_code()

import torchvision

from src.ours.env.creation import (
    PointEnvFactory,
    PointEnvIdentifierGenerator,
    PointEnvConfigFactory,
)
from src.ours.eval.pointenv.expert import PointEnvExpertDefault
from src.ours.util.common.param import PwilParam
from src.ours.util.pwil.manager import (
    PwilManagerFactory,
    PwilManager,
)


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


class PointEnvPwilConfig:
    def __init__(self):
        self._n_demos_pool = [1, 2, 3]
        self._subsampling_pool = [1, 2, 3, 5, 10, 20]

    def _get_configs(self, demo_id_pool: list[int]) -> list[tuple[int, int, int]]:
        configs = []
        for demo_id in demo_id_pool:
            for n_demos in self._n_demos_pool:
                for subsampling in self._subsampling_pool:
                    configs.append((demo_id, n_demos, subsampling))
        return configs

    @staticmethod
    def get_configs():
        demo_id_pool = [0, 1, 2, 3, 4, 5, 6]
        n_demos_pool = [1, 2, 3]
        subsampling_pool = [1, 2, 3, 5, 10, 20]

        configs = []
        for demo_id in demo_id_pool:
            for n_demos in n_demos_pool:
                for subsampling in subsampling_pool:
                    configs.append((demo_id, n_demos, subsampling))
        return configs


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
    trainer = PointEnvPwilManager()
    trainer.test()


if __name__ == "__main__":
    client_code()

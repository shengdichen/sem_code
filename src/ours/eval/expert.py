from gym import Env
from stable_baselines3 import PPO as PPOSB

from src.ours.env.creation import PointEnvFactory, PointEnvIdentifierGenerator
from src.ours.eval.param import ExpertParam
from src.ours.eval.util import SaverManager
from src.ours.util.expert.manager import ExpertManager
from src.ours.util.helper import Plotter, TqdmCallback
from src.ours.util.train import Trainer


class TrainerExpert(Trainer):
    def __init__(self, env: Env, training_param: ExpertParam):
        super().__init__(training_param)

        self._env = env

        self._model = PPOSB(
            "MlpPolicy",
            self._env,
            verbose=0,
            **self._training_param.kwargs_ppo,
            tensorboard_log=self._training_param.sb3_tblog_dir
        )

    @property
    def model(self):
        return self._model

    def train(self) -> None:
        self._model.learn(
            total_timesteps=self._training_param.n_steps_expert_train,
            callback=[TqdmCallback()],
        )


class ExpertClient:
    def __init__(
        self,
        trainer: TrainerExpert,
        managers: tuple[SaverManager, ExpertManager],
        env_identifier: str,
    ):
        self._trainer = trainer
        self._saver_manager, self._expert_manager = managers
        self._env_identifier = env_identifier

    def _train_and_save(self) -> None:
        self._trainer.train()

        self._saver_manager.save(self._env_identifier)
        self._expert_manager.save_expert_traj(self._env_identifier)


class ClientTrainerExpert:
    def __init__(self):
        self._training_param = ExpertParam()
        self._n_timesteps = self._training_param.n_steps_expert_train

    def train_and_plot(self) -> None:
        """
        # Train experts with different shifts representing their waypoint preferences
        """

        for env_config in [
            {"n_targets": 2, "shift_x": 0, "shift_y": 0},
            {"n_targets": 2, "shift_x": 0, "shift_y": 50},
            {"n_targets": 2, "shift_x": 50, "shift_y": 0},
        ]:
            self._train_and_save(env_config)

        self._plot()

    def _train_and_save(self, env_config: dict[str:int]) -> None:
        env = PointEnvFactory(env_config).create()
        trainer = TrainerExpert(env, self._training_param)
        env_identifier = PointEnvIdentifierGenerator(env_config).get_identifier()

        expert_client = ExpertClient(
            trainer,
            (
                SaverManager(trainer.model, self._training_param),
                ExpertManager((env, trainer.model), self._training_param),
            ),
            env_identifier,
        )
        expert_client._train_and_save()

    def _plot(self) -> None:
        Plotter.plot_experts(self._n_timesteps)
        Plotter.plot_experts(self._n_timesteps, hist=False)


def client_code():
    trainer = ClientTrainerExpert()
    trainer.train_and_plot()


if __name__ == "__main__":
    client_code()

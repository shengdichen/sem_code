from src.ours.env.creation import PointEnvFactory, PointEnvIdentifierGenerator
from src.ours.eval.param import ExpertParam
from src.ours.eval.util import Sb3Manager
from src.ours.util.expert.client import ClientExpert
from src.ours.util.expert.manager import ExpertManager
from src.ours.util.expert.train import TrainerExpert
from src.ours.util.helper import Plotter


class PointEnvExpert:
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

        expert_client = ClientExpert(
            trainer,
            (
                Sb3Manager(trainer.model, self._training_param),
                ExpertManager((env, trainer.model), self._training_param),
            ),
            env_identifier,
        )
        expert_client.train()
        expert_client.save()

    def _plot(self) -> None:
        Plotter.plot_experts(self._n_timesteps)
        Plotter.plot_experts(self._n_timesteps, hist=False)


def client_code():
    trainer = PointEnvExpert()
    trainer.train_and_plot()


if __name__ == "__main__":
    client_code()

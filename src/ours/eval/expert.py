from src.ours.env.creation import PointEnvFactory
from src.ours.eval.param import TrainingParam
from src.ours.util.helper import Plotter
from src.ours.util.train import TrainerExpert


class ClientTrainerExpert:
    def __init__(self):
        self._training_param = TrainingParam()
        self._n_timesteps = int(3e5)

    def train_and_plot(self) -> None:
        """
        # Train experts with different shifts representing their waypoint preferences
        """

        self._train({"n_targets": 2, "shift_x": 0, "shift_y": 0}, "exp_0_0")
        self._train({"n_targets": 2, "shift_x": 0, "shift_y": 50}, "exp_0_50")
        self._train({"n_targets": 2, "shift_x": 50, "shift_y": 0}, "exp_50_0")

        self._plot()

    def _train(self, env_config: dict[str:int], fname: str) -> None:
        env = PointEnvFactory(env_config).create()
        trainer = TrainerExpert(self._training_param, env)
        trainer.train(self._n_timesteps, n_targets, shift_x, shift_y, fname)

    def _plot(self) -> None:
        Plotter.plot_experts(self._n_timesteps)
        Plotter.plot_experts(self._n_timesteps, hist=False)


def client_code():
    trainer = ClientTrainerExpert()
    trainer.train_and_plot()


if __name__ == "__main__":
    client_code()

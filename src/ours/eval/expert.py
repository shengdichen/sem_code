from src.ours.env.creation import PointEnvFactory
from src.ours.eval.param import ExpertParam
from src.ours.util.helper import Plotter
from src.ours.util.train import TrainerExpert


class ClientTrainerExpert:
    def __init__(self):
        self._training_param = ExpertParam()
        self._n_timesteps = int(3e5)

    def train_and_plot(self) -> None:
        """
        # Train experts with different shifts representing their waypoint preferences
        """

        self._train({"n_targets": 2, "shift_x": 0, "shift_y": 0})
        self._train({"n_targets": 2, "shift_x": 0, "shift_y": 50})
        self._train({"n_targets": 2, "shift_x": 50, "shift_y": 0})

        self._plot()

    def _train(self, env_config: dict[str:int]) -> None:
        env = PointEnvFactory(env_config).create()
        trainer = TrainerExpert(self._training_param, env)
        filename = self._get_filename_from_shift_values(env_config)

        trainer.train(self._n_timesteps, filename)

    @staticmethod
    def _get_filename_from_shift_values(env_conig: dict[str:int]) -> str:
        shift_x, shift_y = env_conig["shift_x"], env_conig["shift_y"]
        return "exp_" + str(shift_x) + "_" + str(shift_y)

    def _plot(self) -> None:
        Plotter.plot_experts(self._n_timesteps)
        Plotter.plot_experts(self._n_timesteps, hist=False)


def client_code():
    trainer = ClientTrainerExpert()
    trainer.train_and_plot()


if __name__ == "__main__":
    client_code()

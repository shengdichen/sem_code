from src.ours.eval.param import TrainingParam
from src.ours.util.helper import Plotter
from src.ours.util.train import TrainerExpert


class ClientTrainerExpert:
    def __init__(self):
        self._training_param = TrainingParam()
        self._trainer = TrainerExpert(self._training_param)

    def train_experts(self):
        """
        # Train experts with different shifts representing their waypoint preferences
        """
        n_timesteps = 3e5

        self._trainer.train(n_timesteps, 2, 0, 0, fname="exp_0_0")
        self._trainer.train(n_timesteps, 2, 0, 50, fname="exp_0_50")
        self._trainer.train(n_timesteps, 2, 50, 0, fname="exp_50_0")
        Plotter.plot_experts(n_timesteps)

    @staticmethod
    def plot_experts():
        Plotter.plot_experts(5e5)
        Plotter.plot_experts(5e5, hist=False)

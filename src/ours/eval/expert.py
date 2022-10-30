from src.ours.eval.param import TrainingParam
from src.ours.util.helper import Plotter
from src.ours.util.train import train_expert


class TrainerExpert:
    def __init__(self):
        self._train_experts = False
        self._training_param = TrainingParam()

    def train_experts(self):
        """
        # Train experts with different shifts representing their waypoint preferences
        """
        if self._train_experts:
            n_timesteps = 3e5

            model00 = train_expert(
                n_timesteps, 2, 0, 0, self._training_param, fname="exp_0_0"
            )
            model01 = train_expert(
                n_timesteps, 2, 0, 50, self._training_param, fname="exp_0_50"
            )
            model10 = train_expert(
                n_timesteps, 2, 50, 0, self._training_param, fname="exp_50_0"
            )
            Plotter.plot_experts(n_timesteps)

    def plot_experts(self):
        Plotter.plot_experts(5e5)
        Plotter.plot_experts(5e5, hist=False)

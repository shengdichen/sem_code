from src.ours.eval.param import kwargs
from src.ours.util.helper import Plotter
from src.ours.util.train import train_expert

# Train experts with different shifts representing their waypoint preferences
train_experts = False
if train_experts:
    n_timesteps = 3e5

    model00 = train_expert(n_timesteps, 2, 0, 0, kwargs, fname="exp_0_0")
    model01 = train_expert(n_timesteps, 2, 0, 50, kwargs, fname="exp_0_50")
    model10 = train_expert(n_timesteps, 2, 50, 0, kwargs, fname="exp_50_0")
    Plotter.plot_experts(n_timesteps)

Plotter.plot_experts(5e5)
Plotter.plot_experts(5e5, hist=False)

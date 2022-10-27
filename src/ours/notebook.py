import os
import random
import sys
from collections import deque
from datetime import datetime
from itertools import count

# %matplotlib inline
import PIL.Image as Image
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import Env, spaces
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)
from irl import (
    AIRLDiscriminator,
    SWILDiscriminator,
    GAILDiscriminator,
    VAILDiscriminator,
    MEIRLDiscriminator,
    WAILDiscriminator,
)

from utils import prepare_update_airl, CustomCallback
from env_utils import repack_vecenv, PWILReward

# stable baselines imports
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, EvalCallback
from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO as PPOSB


# Test environment
# Define helper functions

### EVAL functions


def test_policy(
    fname,
    model=None,
    rm="ERM",
    shift_x=0,
    shift_y=0,
    n_timesteps=2000,
    deterministic=True,
):
    if model is None:
        model = PPOSB.load(fname)
    testing_env = MovePoint(2, shift_x=shift_x, shift_y=shift_y)

    obs_list = []
    obs = testing_env.reset()
    rew = 0
    cum_rew = []

    for i in range(n_timesteps):
        action, _states = model.predict(obs, deterministic=deterministic)
        obs, r, done, info = testing_env.step(action)
        rew += r
        if done:
            obs = testing_env.reset()
            print(rm + " rewards: ", rew)
            cum_rew.append(rew)

            rew_erm = 0

        obs_list.append(obs)

    print(rm + " mean / std cumrew: ", np.mean(cum_rew), np.std(cum_rew))
    obsa = np.stack(obs_list)
    x, y, bins = get_hist_data(obsa)

    fig = plt.figure()
    plt.title(rm)
    # plt.hist2d(x,y,bins)
    plt.plot(obsa[:, 0], obsa[:, 1], "m-", alpha=0.3)
    plt.scatter(obsa[:, 2], obsa[:, 3], c="r")
    plt.axis("equal")

    return fig


# SETTINGS
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

log_path = "./pointmaze_results"

kwargs = {
    "learning_rate": 0.0003,
    "n_steps": 256,
    "batch_size": 64,
    "n_epochs": 3,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    #'max_grad_norm':0.5
}


# Train experts with different shifts representing their waypoint preferences
train_experts = False
if train_experts:
    n_timesteps = 3e5

    model00 = train_expert(n_timesteps, 2, 0, 0, kwargs, fname="exp_0_0")
    model01 = train_expert(n_timesteps, 2, 0, 50, kwargs, fname="exp_0_50")
    model10 = train_expert(n_timesteps, 2, 50, 0, kwargs, fname="exp_50_0")
    plot_experts(n_timesteps)

plot_experts(5e5)
plot_experts(5e5, hist=False)


# train imitation learning / IRL policy
train_pwil_ = True
if train_pwil_:
    demos = load_expert_demos(5e5)
    flat_demos = [item for sublist in demos for item in sublist]
    model_pwil, plot = train_pwil(
        flat_demos,
        n_demos=3,
        subsampling=10,
        use_actions=False,
        n_timesteps=1e3,
        n_targets=2,
        shift_x=0,
        shift_y=0,
        kwargs=kwargs,
        fname="pwil_0",
        model_dir="./models",
        save_deterministic=False,
    )
    test_policy("", model=model_pwil)
    im = Image.fromarray(plot)
    im.save("pwil.png")
    plt.figure()
    plt.imshow(im)


# plot grid of PWIL rewards
plots = []
demos = load_expert_demos(5e5)
flat_demos_0 = [item for sublist in demos for item in sublist]
flat_demos_01 = [item for sublist in demos[:1] for item in sublist]
flat_demos_12 = [item for sublist in demos[1:] for item in sublist]

for ss in [1, 2, 3, 5, 10, 20]:
    for j, dem in enumerate([demos[0], flat_demos_01, flat_demos_12, flat_demos_0]):
        for n_demos in [1, 2, 3]:

            print("subsampling: ", ss, " dem: ", j, " n_demos: ", n_demos)
            env = PWILReward(
                env=MovePoint(2, 0, 0),
                demos=dem,
                n_demos=n_demos,
                subsampling=ss,
                use_actions=False,
            )
            plots.append(plot_reward(discriminator=None, env=env))
            im = Image.fromarray(plot)
            im.save(
                "pwil_plots/pwil_ss{}_demoidx{}_n_demos{}.png".format(ss, j, n_demos)
            )


# vutils.save_image(plots, normalize=True, nrow=6)


# test_policy('', model=model_pwil)


from argparse import Namespace

SEED = 42

# training settings
env_kwargs = {"n_targets": 2, "shift_x": 0, "shift_y": 0}

kwargs = {
    "learning_rate": 0.0003,
    "n_steps": 256,
    "batch_size": 64,
    "n_epochs": 3,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    #'max_grad_norm':0.5
}

irl_args = {
    "expert_demo_ts": 5e5,
    "use_sb_ppo": True,
    "n_irl_epochs": 5,
    "n_irl_episodes": 100,
    "d_layer_dims": [512, 512],
    "batch_size": 64,
    "latent_dim": 2,
    "lr": 1e-4,
    "gamma": 0.99,
    "use_actions": False,
    "irm_coeff": 0.0,
    "lip_coeff": 0.0,
    "l2_coeff": 0.0,
    # WAIL
    "wail_epsilon": 0.01,
    # VAIL
    "i_c": 1.0,
    "vampprior": True,
    "train_discriminator": True,
    "discriminator_type": "swil",
    "clamp_magnitude": 1000,
    # SWIL
    "n_proj": 100,
    "irl_reward_model": "",
    "resume": None,
}


opt_policy = Namespace(**kwargs)
opt = Namespace(**irl_args)

plots, best_rew_plots, best_val_plots = train_irl(opt, opt_policy, seed=SEED)

imageio.mimsave("rewards_all_" + opt.discriminator_type + ".gif", plots)
imageio.mimsave("rewards_" + opt.discriminator_type + ".gif", best_rew_plots)
imageio.mimsave("values_" + opt.discriminator_type + ".gif", best_val_plots)


opt.irm_coeff = 100.0
plots, best_rew_plots, best_val_plots = train_irl(opt, opt_policy, seed=SEED)

opt.irm_coeff = 10.0
train_irl(opt, opt_policy, seed=SEED)

opt.irm_coeff = 0.1
train_irl(opt, opt_policy, seed=SEED)

opt.irm_coeff = 1.0
train_irl(opt, opt_policy, seed=SEED)


# train on recovered reward
opt.irl_reward_model = "./logs/results_2dnav_irl20220803_232604_train_disc/best_disc.th"
opt.train_discriminator = False

train_irl(opt, opt_policy, seed=SEED)


opt.irl_reward_model = (
    "./logs/results_2dnav_irl20220803_232801irm_1.0_train_disc/disc.th"
)
opt.train_discriminator = False

train_irl(opt, opt_policy, seed=SEED)


fig = test_policy("logs/results_2dnav_irl20220803_233027/best_model.zip", rm="ERM")


fig = test_policy("logs/results_2dnav_irl20220803_234849/best_model.zip", rm="IRM")

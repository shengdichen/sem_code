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
def train_expert(
    n_timesteps,
    n_targets,
    shift_x,
    shift_y,
    kwargs,
    fname,
    model_dir="./models",
    save_deterministic=False,
):
    env = MovePoint(n_targets, shift_x, shift_y)
    model = PPOSB("MlpPolicy", env, verbose=0, **kwargs, tensorboard_log=log_path)
    model.learn(total_timesteps=n_timesteps, callback=[TqdmCallback()])

    # save model
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    model.save(os.path.join(model_dir, "model_" + fname + str(n_timesteps)))
    save_expert_traj(
        env,
        model,
        nr_trajectories=10,
        render=False,
        filename=fname + str(n_timesteps),
        deterministic=save_deterministic,
    )

    return model


def train_pwil(
    demos,
    n_demos,
    subsampling,
    use_actions,
    n_timesteps,
    n_targets,
    shift_x,
    shift_y,
    kwargs,
    fname,
    model_dir="./models",
    save_deterministic=False,
):
    env = PWILReward(
        env=MovePoint(n_targets, shift_x, shift_y),
        demos=demos,
        n_demos=n_demos,
        subsampling=subsampling,
        use_actions=use_actions,
    )

    plot = plot_reward(discriminator=None, env=env)

    testing_env = MovePoint(n_targets, shift_x, shift_y)
    model = PPOSB("MlpPolicy", env, verbose=0, **kwargs, tensorboard_log=log_path)

    eval_callback = EvalCallback(
        testing_env,
        best_model_save_path=log_path,
        log_path=log_path,
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    # eval_callback.init_callback(ppo_dict[k])
    callback_list = CallbackList(
        [CustomCallback(id="", log_path=log_path), eval_callback, TqdmCallback()]
    )

    model.learn(total_timesteps=n_timesteps, callback=callback_list)

    # save model
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    model.save(os.path.join(model_dir, "model_" + fname + str(n_timesteps)))
    save_expert_traj(
        env,
        model,
        nr_trajectories=10,
        render=False,
        filename=fname + str(n_timesteps),
        deterministic=save_deterministic,
    )

    return model, plot


def train_irl(opt, opt_policy, seed):
    # create log dir
    if opt.irm_coeff > 0 and opt.train_discriminator:
        log_suffix = "irm_" + str(opt.irm_coeff)
    else:
        log_suffix = ""

    if opt.train_discriminator:
        log_suffix += "_train_disc"

    log_path = (
        "./logs/results_2dnav_irl"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + log_suffix
    )
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    summary_writer = SummaryWriter(log_path)

    # fix seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # define environments and load expert demos
    env = make_vec_env(MovePoint, n_envs=1, env_kwargs=env_kwargs)
    testing_env = MovePoint(2, shift_x=0, shift_y=0)
    expert_demos = load_expert_demos(opt.expert_demo_ts)

    # define discriminator
    if opt.discriminator_type == "airl":
        discriminator = AIRLDiscriminator(
            env,
            opt.d_layer_dims,
            lr=opt.lr,
            gamma=opt.gamma,
            use_actions=opt.use_actions,
            irm_coeff=opt.irm_coeff,
            lip_coeff=opt.lip_coeff,
            l2_coeff=opt.l2_coeff,
            use_cnn_base=False,
            nonlin=torch.nn.ReLU(),
        )
    elif opt.discriminator_type == "swil":
        discriminator = SWILDiscriminator(
            env,
            opt.d_layer_dims,
            lr=opt.lr,
            batch_size=opt.batch_size,
            use_actions=opt.use_actions,
            n_proj=opt.n_proj,
        )
    elif opt.discriminator_type == "vail":
        discriminator = VAILDiscriminator(
            env,
            opt.d_layer_dims,
            lr=opt.lr,
            latent_dim=opt.latent_dim,
            use_actions=opt.use_actions,
            irm_coeff=opt.irm_coeff,
            lip_coeff=opt.lip_coeff,
            l2_coeff=opt.l2_coeff,
            use_vampprior=opt.vampprior,
            vae_type="TwoLayer",
            i_c=opt.i_c,
            use_cnn_base=False,
        )
    elif opt.discriminator_type == "gail":
        discriminator = GAILDiscriminator(
            env,
            opt.d_layer_dims,
            lr=opt.lr,
            irm_coeff=opt.irm_coeff,
            lip_coeff=opt.lip_coeff,
            l2_coeff=opt.l2_coeff,
            use_actions=opt.use_actions,
        )
    elif opt.discriminator_type == "wail":
        discriminator = WAILDiscriminator(
            env,
            opt.d_layer_dims,
            lr=opt.lr,
            use_actions=opt.use_actions,
            irm_coeff=opt.irm_coeff,
            lip_coeff=opt.lip_coeff,
            l2_coeff=opt.l2_coeff,
            epsilon=opt.wail_epsilon,
        )
    elif opt.discriminator_type == "meirl":
        discriminator = MEIRLDiscriminator(
            env,
            opt.d_layer_dims,
            lr=opt.lr,
            clamp_magnitude=opt.clamp_magnitude,
            irm_coeff=opt.irm_coeff,
            lip_coeff=opt.lip_coeff,
            l2_coeff=opt.l2_coeff,
            use_actions=opt.use_actions,
        )

    # if we're training with the learned reward, show what we're training with
    if not opt.train_discriminator:
        discriminator.load_state_dict(torch.load(opt.irl_reward_model))
        plot_reward(discriminator)
        if opt.discriminator_type == "airl":
            plot_reward(discriminator, plot_value=True)

    # and wrap environment with irl reward
    env = repack_vecenv(env, disc=discriminator)

    # define imitation policy with respective callbacks
    policy = PPOSB("MlpPolicy", env, **kwargs, tensorboard_log=log_path)
    new_logger = configure_logger(tensorboard_log=log_path)
    policy.ep_info_buffer = deque(maxlen=100)
    policy.ep_success_buffer = deque(maxlen=100)
    policy.set_logger(new_logger)

    if opt.resume is not None:
        policy.load(os.path.join(opt.resume, "best_model.zip"), env)

    ## create callbacks to evaluate and plot ground truth reward
    if opt.train_discriminator:
        callback_on_best = RewardCheckpointCallback(
            discriminator=discriminator,
            verbose=1,
            log_path=log_path,
            plot_value=(opt.discriminator_type == "airl"),
        )
    else:
        callback_on_best = None
    eval_callback = EvalCallback(
        testing_env,
        best_model_save_path=log_path,
        callback_on_new_best=callback_on_best,
        log_path=log_path,
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    # eval_callback.init_callback(ppo_dict[k])
    callback_list = CallbackList(
        [CustomCallback(id="", log_path=log_path), eval_callback]
    )
    callback_list.init_callback(policy)

    total_numsteps = 0
    losses = []
    plot_list = []

    for i_update in tqdm(range(opt.n_irl_episodes), "episodes"):
        ## collect rollout buffer
        if policy._last_obs is None:
            policy._last_obs = env.reset()
        policy.collect_rollouts(
            env,
            callback_list,
            policy.rollout_buffer,
            n_rollout_steps=opt_policy.n_steps,
        )

        total_numsteps = total_numsteps + opt_policy.n_steps

        ## train policy
        policy.train()
        # different buffer spec in SB
        policy.logger.dump(total_numsteps)

        # update discriminator
        if opt.train_discriminator:
            for irl_epoch in range(opt.n_irl_epochs):
                transitions = next(policy.rollout_buffer.get(opt_policy.batch_size))
                policy_state_batch = transitions.observations
                policy_action_batch = transitions.actions

                bce_losses = {}
                policy_estimates = {}
                expert_estimates = {}
                grad_pens = {}
                for i, demo in enumerate(expert_demos):
                    update_dict = prepare_update_airl(
                        env, opt, demo, policy_state_batch, policy_action_batch, policy
                    )

                    output_dict = discriminator.compute_loss(update_dict)
                    bce_losses[i] = output_dict["d_loss"]
                    # policy_estimates[i] = output_dict['policy_estimate']
                    # expert_estimates[i] = output_dict['expert_estimate']
                    grad_pens[i] = output_dict["grad_penalty"]

                bce_loss_all = torch.stack(list(bce_losses.values())).mean()
                losses.append(bce_loss_all.detach().numpy())
                # policy_estimates_all = torch.stack(list(policy_estimates.values())).mean()
                # expert_estimates_all = torch.stack(list(expert_estimates.values())).mean()
                grad_pen_all = torch.stack(list(grad_pens.values())).mean()
                loss = bce_loss_all + opt.irm_coeff * grad_pen_all
                # TODO: is this necessary?
                if opt.irm_coeff > 1.0:
                    loss /= opt.irm_coeff

                discriminator.update(loss)

                # summary_writer.add_scalar('IRL/AIRL_policy_estimate', policy_estimates_all, i_update)
                # summary_writer.add_scalar('IRL/'AIRL_expert_estimate', expert_estimates_all, i_update)
                summary_writer.add_scalar(
                    "IRL/" + opt.discriminator_type + "_bceloss", bce_loss_all, i_update
                )
                if opt.irm_coeff > 0:
                    summary_writer.add_scalar(
                        "IRL/" + opt.discriminator_type + "_irmloss",
                        grad_pen_all,
                        i_update,
                    )

                if i_update % 1000 == 0:
                    print("Reward at iteration: ", i_update)
                    if opt.discriminator_type == "airl":
                        plot_reward(discriminator, plot_value=True)

            torch.save(discriminator.state_dict(), os.path.join(log_path, "disc.th"))
            plot_list.append(plot_reward(discriminator))
            if opt.discriminator_type == "airl":
                plot_reward(discriminator, plot_value=True)

    plt.plot(losses)

    if callback_on_best is not None:
        return (
            plot_list,
            callback_on_best.plot_list_reward,
            callback_on_best.plot_list_value,
        )
    else:
        return [], [], []


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

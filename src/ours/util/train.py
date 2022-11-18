import os
import random
from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import Env
from stable_baselines3 import PPO as PPOSB
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import configure_logger
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.ours.env.env import MovePoint
from src.ours.eval.param import TrainingParam
from src.ours.util.helper import (
    TqdmCallback,
    ExpertManager,
    RewardPlotter,
    RewardCheckpointCallback,
)
from src.upstream.env_utils import PWILReward, repack_vecenv
from src.upstream.irl import (
    AIRLDiscriminator,
    SWILDiscriminator,
    GAILDiscriminator,
    VAILDiscriminator,
    MEIRLDiscriminator,
    WAILDiscriminator,
)
from src.upstream.utils import CustomCallback, prepare_update_airl


class Trainer:
    def __init__(self, training_param: TrainingParam):
        self._training_param = training_param
        self._log_path = self._training_param.log_path
        self._kwargs_ppo = self._training_param.kwargs_ppo

    def train(self, **kwargs):
        pass


class TrainerExpert(Trainer):
    def __init__(self, training_param: TrainingParam, env: Env):
        super().__init__(training_param)

        self._env = env
        self._model_dir = "./models"
        self._save_deterministic = False

    def train(self, n_timesteps, fname):
        model = PPOSB(
            "MlpPolicy",
            self._env,
            verbose=0,
            **self._kwargs_ppo,
            tensorboard_log=self._log_path
        )
        model.learn(total_timesteps=n_timesteps, callback=[TqdmCallback()])

        # save model
        if not os.path.exists(self._model_dir):
            os.mkdir(self._model_dir)

        model.save(os.path.join(self._model_dir, "model_" + fname + str(n_timesteps)))
        ExpertManager.save_expert_traj(
            self._env,
            model,
            nr_trajectories=10,
            render=False,
            filename=fname + str(n_timesteps),
            deterministic=self._save_deterministic,
        )

        return model


class TrainerPwil(Trainer):
    def __init__(self, training_param: TrainingParam):
        super().__init__(training_param)

        self._model_dir = "./models"
        self._save_deterministic = False

    def train(
        self,
        demos,
        n_demos,
        subsampling,
        use_actions,
        n_timesteps,
        n_targets,
        shift_x,
        shift_y,
        fname,
    ):
        env = PWILReward(
            env=MovePoint(n_targets, shift_x, shift_y),
            demos=demos,
            n_demos=n_demos,
            subsampling=subsampling,
            use_actions=use_actions,
        )

        plot = RewardPlotter.plot_reward(discriminator=None, env=env)

        testing_env = MovePoint(n_targets, shift_x, shift_y)
        model = PPOSB(
            "MlpPolicy",
            env,
            verbose=0,
            **self._kwargs_ppo,
            tensorboard_log=self._log_path
        )

        eval_callback = EvalCallback(
            testing_env,
            best_model_save_path=self._log_path,
            log_path=self._log_path,
            eval_freq=10000,
            deterministic=True,
            render=False,
        )

        # eval_callback.init_callback(ppo_dict[k])
        callback_list = CallbackList(
            [
                CustomCallback(id="", log_path=self._log_path),
                eval_callback,
                TqdmCallback(),
            ]
        )

        model.learn(total_timesteps=n_timesteps, callback=callback_list)

        # save model
        if not os.path.exists(self._model_dir):
            os.mkdir(self._model_dir)

        model.save(os.path.join(self._model_dir, "model_" + fname + str(n_timesteps)))
        ExpertManager.save_expert_traj(
            env,
            model,
            nr_trajectories=10,
            render=False,
            filename=fname + str(n_timesteps),
            deterministic=self._save_deterministic,
        )

        return model, plot


class TrainerIrl(Trainer):
    def train(self, opt, opt_policy, seed, env_kwargs):
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
        expert_demos = ExpertManager.load_expert_demos(opt.expert_demo_ts)

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
        else:
            print("Specified discriminator invalid!")
            return 1

        # if we're training with the learned reward, show what we're training with
        if not opt.train_discriminator:
            discriminator.load_state_dict(torch.load(opt.irl_reward_model))
            RewardPlotter.plot_reward(discriminator)
            if opt.discriminator_type == "airl":
                RewardPlotter.plot_reward(discriminator, plot_value=True)

        # and wrap environment with irl reward
        env = repack_vecenv(env, disc=discriminator)

        # define imitation policy with respective callbacks
        policy = PPOSB("MlpPolicy", env, **self._kwargs_ppo, tensorboard_log=log_path)
        new_logger = configure_logger(tensorboard_log=log_path)
        policy.ep_info_buffer = deque(maxlen=100)
        policy.ep_success_buffer = deque(maxlen=100)
        policy.set_logger(new_logger)

        if opt.resume is not None:
            policy.load(os.path.join(opt.resume, "best_model.zip"), env)

        # create callbacks to evaluate and plot ground truth reward
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
            # collect rollout buffer
            if policy._last_obs is None:
                policy._last_obs = env.reset()
            policy.collect_rollouts(
                env,
                callback_list,
                policy.rollout_buffer,
                n_rollout_steps=opt_policy.n_steps,
            )

            total_numsteps = total_numsteps + opt_policy.n_steps

            # train policy
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
                    # policy_estimates = {}
                    # expert_estimates = {}
                    grad_pens = {}
                    for i, demo in enumerate(expert_demos):
                        update_dict = prepare_update_airl(
                            env,
                            opt,
                            demo,
                            policy_state_batch,
                            policy_action_batch,
                            policy,
                        )

                        output_dict = discriminator.compute_loss(update_dict)
                        bce_losses[i] = output_dict["d_loss"]
                        # policy_estimates[i] = output_dict['policy_estimate']
                        # expert_estimates[i] = output_dict['expert_estimate']
                        grad_pens[i] = output_dict["grad_penalty"]

                    bce_loss_all = torch.stack(list(bce_losses.values())).mean()
                    losses.append(bce_loss_all.detach().numpy())
                    # policy_estimates_all = torch.stack(
                    #     list(policy_estimates.values())
                    # ).mean()
                    # expert_estimates_all = torch.stack(
                    #     list(expert_estimates.values())
                    # ).mean()
                    grad_pen_all = torch.stack(list(grad_pens.values())).mean()
                    loss = bce_loss_all + opt.irm_coeff * grad_pen_all
                    # TODO: is this necessary?
                    if opt.irm_coeff > 1.0:
                        loss /= opt.irm_coeff

                    discriminator.update(loss)

                    # summary_writer.add_scalar(
                    #     "IRL/AIRL_policy_estimate", policy_estimates_all, i_update
                    # )
                    # summary_writer.add_scalar(
                    #     "IRL/AIRL_expert_estimate", expert_estimates_all, i_update
                    # )
                    summary_writer.add_scalar(
                        "IRL/" + opt.discriminator_type + "_bceloss",
                        bce_loss_all,
                        i_update,
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
                            RewardPlotter.plot_reward(discriminator, plot_value=True)

                torch.save(
                    discriminator.state_dict(), os.path.join(log_path, "disc.th")
                )
                plot_list.append(RewardPlotter.plot_reward(discriminator))
                if opt.discriminator_type == "airl":
                    RewardPlotter.plot_reward(discriminator, plot_value=True)

        plt.plot(losses)

        if callback_on_best is not None:
            return (
                plot_list,
                callback_on_best.plot_list_reward,
                callback_on_best.plot_list_value,
            )
        else:
            return [], [], []

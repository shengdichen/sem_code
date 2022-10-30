from argparse import Namespace

import imageio

from src.ours.eval.param import TrainingParam
from src.ours.util.test import PolicyTester
from src.ours.util.train import Training


class TrainingIrl:
    def __init__(self):
        self._seed = 42

        self._env_kwargs = {"n_targets": 2, "shift_x": 0, "shift_y": 0}

        self._kwargs = {
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
            # 'max_grad_norm':0.5
        }

        self._irl_args = {
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

        self._opt_policy = Namespace(**self._kwargs)
        self._opt = Namespace(**self._irl_args)

        self._training_param = TrainingParam()
        self._training = Training(self._training_param)

    def training_settings(self):
        plots, best_rew_plots, best_val_plots = self._training.train_irl(
            self._opt, self._opt_policy, self._seed, self._env_kwargs
        )

        imageio.mimsave("rewards_all_" + self._opt.discriminator_type + ".gif", plots)
        imageio.mimsave(
            "rewards_" + self._opt.discriminator_type + ".gif", best_rew_plots
        )
        imageio.mimsave(
            "values_" + self._opt.discriminator_type + ".gif", best_val_plots
        )

    def train_various_irl(self):
        self._opt.irm_coeff = 100.0
        plots, best_rew_plots, best_val_plots = self._training.train_irl(
            self._opt, self._opt_policy, self._seed, self._env_kwargs
        )

        self._opt.irm_coeff = 10.0
        self._training.train_irl(
            self._opt, self._opt_policy, self._seed, self._env_kwargs
        )

        self._opt.irm_coeff = 0.1
        self._training.train_irl(
            self._opt, self._opt_policy, self._seed, self._env_kwargs
        )

        self._opt.irm_coeff = 1.0
        self._training.train_irl(
            self._opt, self._opt_policy, self._seed, self._env_kwargs
        )

    def train_on_reward(self):
        # train on recovered reward
        self._opt.irl_reward_model = (
            "./logs/results_2dnav_irl20220803_232604_train_disc/best_disc.th"
        )
        self._opt.train_discriminator = False

        self._training.train_irl(
            self._opt, self._opt_policy, self._seed, self._env_kwargs
        )

        self._opt.irl_reward_model = (
            "./logs/results_2dnav_irl20220803_232801irm_1.0_train_disc/disc.th"
        )
        self._opt.train_discriminator = False

        self._training.train_irl(
            self._opt, self._opt_policy, self._seed, self._env_kwargs
        )

        fig = PolicyTester.test_policy(
            "logs/results_2dnav_irl20220803_233027/best_model.zip", rm="ERM"
        )

        fig = PolicyTester.test_policy(
            "logs/results_2dnav_irl20220803_234849/best_model.zip", rm="IRM"
        )

from argparse import Namespace

import imageio

from src.ours.eval.param import CommonParam
from src.ours.util.test import PolicyTester
from src.ours.util.train import TrainerIrl


class ClientTrainingIrl:
    def __init__(self):
        # TODO:
        #   use training_param instead of separately naming these things
        self._seed = 42

        self._env_kwargs = {"n_targets": 2, "shift_x": 0, "shift_y": 0}

        self._kwargs_ppo = {
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

        self._kwargs_irl = {
            "expert_demo_ts": int(3e5),
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

        self._opt_ppo = Namespace(**self._kwargs_ppo)
        self._opt_irl = Namespace(**self._kwargs_irl)

        self._training_param = CommonParam()
        self._trainer = TrainerIrl(self._training_param)

    def training_settings(self):
        plots, best_rew_plots, best_val_plots = self._trainer.train(
            self._opt_irl, self._opt_ppo, self._seed, self._env_kwargs
        )

        imageio.mimsave("rewards_all_" + self._opt_irl.discriminator_type + ".gif", plots)
        imageio.mimsave(
            "rewards_" + self._opt_irl.discriminator_type + ".gif", best_rew_plots
        )
        imageio.mimsave(
            "values_" + self._opt_irl.discriminator_type + ".gif", best_val_plots
        )

    def train_various_irl(self):
        for irm_coeff in [100, 10, 0.1, 1.0]:
            self._opt_irl.irm_coeff = irm_coeff
            self._trainer.train(
                self._opt_irl, self._opt_ppo, self._seed, self._env_kwargs
            )

    def train_on_reward(self):
        # train on recovered reward
        self._opt_irl.irl_reward_model = (
            "./logs/results_2dnav_irl20220803_232604_train_disc/best_disc.th"
        )
        self._opt_irl.train_discriminator = False

        self._trainer.train(self._opt_irl, self._opt_ppo, self._seed, self._env_kwargs)

        self._opt_irl.irl_reward_model = (
            "./logs/results_2dnav_irl20220803_232801irm_1.0_train_disc/disc.th"
        )
        self._opt_irl.train_discriminator = False

        self._trainer.train(self._opt_irl, self._opt_ppo, self._seed, self._env_kwargs)

        PolicyTester.test_policy(
            "logs/results_2dnav_irl20220803_233027/best_model.zip", rm="ERM"
        )

        PolicyTester.test_policy(
            "logs/results_2dnav_irl20220803_234849/best_model.zip", rm="IRM"
        )


def client_code():
    trainer = ClientTrainingIrl()
    trainer.training_settings()


if __name__ == "__main__":
    client_code()

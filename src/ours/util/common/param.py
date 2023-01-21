import random
from pathlib import Path

import numpy as np
import torch


class Util:
    @staticmethod
    def mkdir_if_not_existent(directories: list[str]):
        for d in directories:
            Path(d).mkdir(parents=True, exist_ok=True)


class CommonParam:
    def __init__(self):
        self._seed = 42
        self._propagate_seed()

        self._sb3_tblog_dir = "./sb3_tb/"
        Util.mkdir_if_not_existent([self._sb3_tblog_dir])
        self._model_dir, self._demo_dir = "./", "./"

        self._prefix, self._postfix = "exp", "_expert_traj.npy"

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
            # "max_grad_norm":0.5
        }

        self._n_steps_expert_train = int(1e6)

    @property
    def seed(self):
        return self._seed

    @property
    def sb3_tblog_dir(self):
        return self._sb3_tblog_dir

    @property
    def model_dir(self):
        return self._model_dir

    @property
    def demo_dir(self):
        return self._demo_dir

    @property
    def kwargs_ppo(self):
        return self._kwargs_ppo

    @property
    def n_steps_expert_train(self):
        return self._n_steps_expert_train

    @property
    def prefix(self):
        return self._prefix

    @property
    def postfix(self):
        return self._postfix

    def _propagate_seed(self):
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)
        random.seed(self._seed)


class ExpertParam(CommonParam):
    def __init__(self):
        super().__init__()

        self._model_dir, self._demo_dir = "./expert/models/", "./expert/demos/"
        Util.mkdir_if_not_existent([self._model_dir, self._demo_dir])


class PwilParam(CommonParam):
    def __init__(self):
        super().__init__()

        self._model_dir, self._demo_dir, self._rewardplot_dir = (
            "./pwil/models/",
            "./pwil/demos/",
            "./pwil/rewardplot/",
        )
        Util.mkdir_if_not_existent(
            [self._model_dir, self._demo_dir, self._rewardplot_dir]
        )

        self._n_steps_pwil_train = int(5e5)

        self._pwil_training_param = {
            "n_demos": 5,
            "subsampling": 10,
            "use_actions": True,
        }

        self._trajectory_num = 0

    @property
    def rewardplot_dir(self):
        return self._rewardplot_dir

    @property
    def n_steps_pwil_train(self):
        return self._n_steps_pwil_train

    @property
    def pwil_training_param(self):
        return self._pwil_training_param

    @pwil_training_param.setter
    def pwil_training_param(self, value):
        self._pwil_training_param = value

    @property
    def trajectory_num(self):
        return self._trajectory_num

    @trajectory_num.setter
    def trajectory_num(self, value: int):
        self._trajectory_num = value

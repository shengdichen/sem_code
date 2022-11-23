import random
from pathlib import Path

import numpy as np
import torch


class Util:
    @staticmethod
    def mkdir_if_not_existent(directories: list[str]):
        for d in directories:
            Path(d).mkdir(exist_ok=True)


class CommonParam:
    def __init__(self):
        self._seed = 42
        self._propagate_seed()

        self._sb3_tblog_dir = "./pointmaze_results/"
        Util.mkdir_if_not_existent([self._sb3_tblog_dir])
        self._model_dir, self._demo_dir = "./", "./"

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

        self._n_steps_expert_train = int(3e5)

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

    def _propagate_seed(self):
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)
        random.seed(self._seed)


class ExpertParam(CommonParam):
    def __init__(self):
        super().__init__()

        self._model_dir, self._demo_dir = "./models/", "./demos/"
        Util.mkdir_if_not_existent([self._model_dir, self._demo_dir])


class PwilParam(CommonParam):
    def __init__(self):
        super().__init__()

        self._model_dir, self._demo_dir = "./models_pwil/", "./demos/"
        Util.mkdir_if_not_existent([self._model_dir, self._demo_dir])

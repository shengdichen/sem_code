import random

import numpy as np
import torch


class CommonParam:
    def __init__(self):
        self._seed = 42
        self._propagate_seed()

        self._sb3_tblog_dir = "./pointmaze_results"

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

    @property
    def seed(self):
        return self._seed

    @property
    def sb3_tblog_dir(self):
        return self._sb3_tblog_dir

    @property
    def kwargs_ppo(self):
        return self._kwargs_ppo

    def _propagate_seed(self):
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)
        random.seed(self._seed)


class ExpertParam(CommonParam):
    def __init__(self):
        super().__init__()
        self._model_dir = "./models/"

    @property
    def model_dir(self):
        return self._model_dir


class PwilParam(CommonParam):
    def __init__(self):
        super().__init__()
        self._model_dir = "./models_pwil/"

    @property
    def model_dir(self):
        return self._model_dir

import random

import numpy as np
import torch

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

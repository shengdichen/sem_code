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


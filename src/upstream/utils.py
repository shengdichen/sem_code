import torch
import torch.multiprocessing
import torch.nn as nn
import numpy as np
from itertools import count
from datetime import datetime
from typing import Callable, Union
import os
import yaml

# import doorenv
import gym

# from gym_minigrid.wrappers import *
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    CallbackList,
    EvalCallback,
)
from stable_baselines3.common.utils import get_linear_fn, set_random_seed
from stable_baselines3.common.utils import configure_logger
from stable_baselines3 import PPO as PPOSB
from stable_baselines3 import SAC as SACSB

# from stable_baselines3.common.logger import logger
# from stable_baselines3.common.monitor import Monitor

from env_utils import make_venv, format_name_string, make_robosuite_env, pmObsWrapper
from torch.utils.tensorboard import SummaryWriter

from setuptools.command.saveopts import saveopts

# obsolete
# def ppo_iter(mini_batch_size, obs, acs, returns, advantage):
# batch_size = obs.shape[0]
# for _ in range(batch_size // mini_batch_size):
# rand_ids = np.random.randint(0, batch_size, mini_batch_size)
# yield (obs[rand_ids, :], acs[rand_ids, :],
# returns[rand_ids, :], advantage[rand_ids, :])


def prepare_update_airl(env, opt, expert_demos, obs, acs, policy):
    ac_sample = env.action_space.sample()
    if isinstance(ac_sample, int):
        ac_shape = 1
    else:
        ac_shape = ac_sample.shape[-1]

    if isinstance(obs, torch.Tensor):
        obs = obs.cpu().numpy()
    if isinstance(acs, torch.Tensor):
        acs = acs.cpu().numpy()

    # flatten first dimension to use samples from all env
    obs = np.reshape(obs, [-1, obs.shape[-1]])
    acs = np.reshape(acs, [-1, ac_shape])
    obs_next = np.concatenate([obs[1:], np.expand_dims(obs[-1], 0)], axis=0)

    # sample expert_demos
    expert_ob_ac_done_reward = expert_demos[
        np.random.randint(0, expert_demos.shape[0], opt.batch_size), :
    ]
    expert_dones = expert_ob_ac_done_reward[:, -1]
    rewards = expert_ob_ac_done_reward[:, -2]
    expert_ob_ac = expert_ob_ac_done_reward[:, :-2]
    expert_obs = expert_ob_ac[:, :-ac_shape]
    expert_acs = expert_ob_ac[:, -ac_shape:]
    expert_obs_next = np.concatenate(
        [expert_obs[1:], np.expand_dims(expert_obs[-1], 0)], axis=0
    )  # repeat last observation

    # convert to torch tensors
    obs_t = torch.from_numpy(obs).type(torch.get_default_dtype())
    acs_t = torch.from_numpy(acs).type(torch.get_default_dtype())
    expert_obs_t = torch.from_numpy(expert_obs).type(torch.get_default_dtype())
    expert_acs_t = torch.from_numpy(expert_acs).type(torch.get_default_dtype())

    # policy_ob_ac = np.concatenate([obs, acs], 1)
    # eval lprobs conditioned on obs, acs
    with torch.no_grad():
        if opt.use_sb_ppo:
            if torch.cuda.is_available():
                _, _, expert_lprobs_t = policy.policy.forward(expert_obs_t.cuda())
                _, _, policy_lprobs_t = policy.policy.forward(obs_t.cuda())
            else:
                _, _, expert_lprobs_t = policy.policy.forward(expert_obs_t)
                _, _, policy_lprobs_t = policy.policy.forward(obs_t)
        else:
            expert_lprobs_t = policy.get_lprobs(expert_obs_t, expert_acs_t)
            policy_lprobs_t = policy.get_lprobs(obs_t, acs_t)

    all_obs = np.concatenate([expert_obs, obs], axis=0)
    all_next_obs = np.concatenate([expert_obs_next, obs_next], axis=0)
    all_acs = np.concatenate([expert_acs, acs], axis=0)
    lprobs = np.concatenate(
        [expert_lprobs_t.cpu().numpy(), policy_lprobs_t.cpu().numpy()], axis=0
    )

    expert_obs_t = torch.from_numpy(expert_obs).type(torch.get_default_dtype())
    expert_acs_t = torch.from_numpy(expert_acs).type(torch.get_default_dtype())
    expert_obs_next_t = torch.from_numpy(expert_obs_next).type(
        torch.get_default_dtype()
    )
    expert_dones_t = torch.from_numpy(expert_dones).type(torch.get_default_dtype())
    policy_obs_t = torch.from_numpy(obs).type(torch.get_default_dtype())
    policy_acs_t = torch.from_numpy(acs).type(torch.get_default_dtype())
    policy_obs_next_t = torch.from_numpy(obs_next).type(torch.get_default_dtype())
    all_obs_t = torch.from_numpy(all_obs).type(torch.get_default_dtype())
    all_obs_next_t = torch.from_numpy(all_next_obs).type(torch.get_default_dtype())
    all_acs_t = torch.from_numpy(all_acs).type(torch.get_default_dtype())
    all_lprobs_t = torch.from_numpy(lprobs).type(torch.get_default_dtype())

    if torch.cuda.is_available():
        expert_obs_t = expert_obs_t.cuda()
        expert_obs_next_t = expert_obs_next_t.cuda()
        expert_acs_t = expert_acs_t.cuda()
        expert_lprobs_t = expert_lprobs_t.cuda()
        expert_dones_t = expert_dones_t.cuda()
        policy_obs_next_t = policy_obs_next_t.cuda()
        policy_obs_t = policy_obs_t.cuda()
        policy_acs_t = policy_acs_t.cuda()
        policy_lprobs_t = policy_lprobs_t.cuda()

        all_obs_next_t = all_obs_next_t.cuda()
        all_obs_t = all_obs_t.cuda()
        all_acs_t = all_acs_t.cuda()
        all_lprobs_t = all_lprobs_t.cuda()

    update_dict = {}
    update_dict["expert_obs"] = expert_obs_t
    update_dict["expert_obs_next"] = expert_obs_next_t
    update_dict["expert_acs"] = expert_acs_t
    update_dict["expert_lprobs"] = expert_lprobs_t
    update_dict["expert_dones"] = expert_dones_t

    update_dict["policy_obs"] = policy_obs_t
    update_dict["policy_obs_next"] = policy_obs_next_t
    update_dict["policy_acs"] = policy_acs_t
    update_dict["policy_lprobs"] = policy_lprobs_t

    update_dict["all_obs"] = all_obs_t
    update_dict["all_obs_next"] = all_obs_next_t
    update_dict["all_acs"] = all_acs_t
    update_dict["all_lprobs"] = all_lprobs_t

    return update_dict


def log_Normal_diag(x, mean, log_var, average=False, dim=None):
    log_normal = -0.5 * (log_var + torch.pow(x - mean, 2) / torch.exp(log_var))
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)


def log_Normal_standard(x, average=False, dim=None):
    log_normal = -0.5 * torch.pow(x, 2)
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)


def vampprior_kld_vae(sample_dict, n_pseudo_inputs):
    mu = sample_dict["z1_mu"]
    logvar = sample_dict["z1_logvar"]
    z = sample_dict["z1"]
    z_p_mu = sample_dict["z1_p_mu"]
    z_p_logvar = sample_dict["z1_p_logvar"]

    z_expanded = z.unsqueeze(1)
    means = z_p_mu.unsqueeze(0)
    logvars = z_p_logvar.unsqueeze(0)
    log_normal = log_Normal_diag(z_expanded, means, logvars, dim=2) - np.log(
        n_pseudo_inputs
    )
    # if args.attention and args.attention_type == 'nn':
    #    log_normal = vdb.attention_weights * log_normal
    log_normal_max, _ = torch.max(log_normal, 1)
    log_p_z = log_normal_max + torch.log(
        torch.sum(torch.exp(log_normal - log_normal_max.unsqueeze(1)), 1)
    )
    log_q_z = log_Normal_diag(z, mu, logvar, dim=1)
    kld = -(log_p_z - log_q_z)
    kld = kld.mean()
    return kld


def vampprior_kld_twolayervae(sample_dict, n_pseudo_inputs, use_vampprior=True):
    log_p_z1 = log_Normal_diag(
        sample_dict["z1"], sample_dict["z1_p_mu"], sample_dict["z1_p_logvar"], dim=1
    )
    log_q_z1 = log_Normal_diag(
        sample_dict["z1"], sample_dict["z1_mu"], sample_dict["z1_logvar"], dim=1
    )
    log_q_z2 = log_Normal_diag(
        sample_dict["z2"], sample_dict["z2_mu"], sample_dict["z2_logvar"], dim=1
    )
    if use_vampprior:
        z_expanded = sample_dict["z2"].unsqueeze(1)
        means = sample_dict["z2_p_mu"].unsqueeze(0)
        logvars = sample_dict["z2_p_logvar"].unsqueeze(0)
        log_normal = log_Normal_diag(z_expanded, means, logvars, dim=2) - np.log(
            n_pseudo_inputs
        )
        log_normal_max, _ = torch.max(log_normal, 1)
        log_p_z2 = log_normal_max + torch.log(
            torch.sum(torch.exp(log_normal - log_normal_max.unsqueeze(1)), 1)
        )
    else:
        log_p_z2 = log_Normal_standard(sample_dict["z2"], dim=1)
    kld = -(log_p_z1 + log_p_z2 - log_q_z1 - log_q_z2)
    kld = kld.mean()
    return kld


def gaussian_kld(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1)


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, verbose=0, id="", log_path=None):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.cnt = 0
        self.id = id
        self.episode_reward = 0
        self.episode_reward_gt = 0
        self.aux_writer = SummaryWriter(log_path)

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        self.episode_reward = 0
        self.episode_reward_gt = 0

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def on_step(self) -> bool:
        """
        This method will be called by the model after each call to ``env.step()``.

        For child callback (of an ``EventCallback``), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        self.n_calls += 1
        rews = self.locals["rewards"]
        infos = self.locals["infos"]
        rew = np.mean(rews)
        rew_gt = np.mean(
            [inf["gt_reward"] if "gt_reward" in inf.keys() else 0 for inf in infos]
        )
        self.episode_reward += rew
        self.episode_reward_gt += rew_gt
        self.cnt += 1

        return self._on_step()

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.aux_writer.add_scalar(
            "Reward/Ep_rewards_" + self.id, self.episode_reward, self.cnt
        )
        self.aux_writer.add_scalar(
            "Reward/Ep_rewards_gt_" + self.id, self.episode_reward_gt, self.cnt
        )

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

import torch
import torch.multiprocessing
import torch.nn as nn
import numpy as np
from itertools import count
from datetime import datetime
from typing import Callable, Union
import os
import yaml
#import doorenv
import gym
#from gym_minigrid.wrappers import *
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList, EvalCallback
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


def dump(obj):
    for attr in dir(obj):
        print("obj.%s = %r" % (attr, getattr(obj, attr)))


def test_env(env, model, vis=False):
    ob = env.reset()
    done = False
    total_reward = 0
    while not done:
        if vis:
            env.render()
        act = model.get_action(torch.from_numpy(ob).type(torch.get_default_dtype()))
        next_ob, reward, done, _ = env.step(act.detach().numpy())
        ob = next_ob
        total_reward += reward
    return total_reward


def eval_render(env, model, num_steps):
    ob = env.reset()
    for i in range(num_steps):
        # ac = ppo.get_action(np.expand_dims(ob,0))
        ac = model.get_action(torch.from_numpy(ob).type(torch.get_default_dtype()))[0]
        act = model.get_action(torch.from_numpy(ob).type(torch.get_default_dtype()))
        next_ob, reward, done, _ = env.step([ac.detach().item()])
        # next_ob, _, done, _ = env.step(ac[0])
        env.render()

    env.close()


def save_expert_traj(opt, env, model, spec_kwargs, extra_reward_threshold=0,
                     nr_trajectories=10, stable_baselines_model=False, pl_model_file=None):
    num_steps = 0
    expert_traj = []
    expert_traj_extra = []

    if isinstance(nr_trajectories, list):
        nr_trajectories = nr_trajectories[0]

    if pl_model_file is not None:
        print(pl_model_file)
    for i_episode in count():
        ob = env.reset()
        done = False
        total_reward = 0
        episode_traj = []

        while not done:
            if stable_baselines_model:
                ac, _states = model.predict(ob)
                # print(env)
                # if not isinstance(ac,list):
                # ac = np.array([ac])
                next_ob, reward, done, _ = env.step(ac)
            else:
                ac = model.get_action(torch.from_numpy(ob).type(torch.get_default_dtype()))
                if not isinstance(env.action_space, gym.spaces.Discrete):
                    ac = [ac.detach().item()]
                else:
                    ac = ac.detach().item()
                next_ob, reward, done, _ = env.step(ac)
            ob = next_ob
            total_reward += reward
            # if len(ob.shape) != len(ac.shape):
            # print("shape mismatch")
            # ob = np.squeeze(ob)
            stacked_vec = np.hstack([np.squeeze(ob), np.squeeze(ac), reward, done])
            expert_traj.append(stacked_vec)
            episode_traj.append(stacked_vec)
            num_steps += 1

        print("episode:", i_episode, "reward:", total_reward,
                "extra threshold", extra_reward_threshold)

        if total_reward > extra_reward_threshold:
            expert_traj_extra.extend(episode_traj)

        if i_episode >= nr_trajectories:
            break

    filename = opt.env_name + format_name_string(str(spec_kwargs))
    if pl_model_file is not None:
        filename = filename + pl_model_file

    if not os.path.exists(opt.demo_dir):
        os.mkdir(opt.demo_dir)
        os.mkdir(os.path.join(opt.demo_dir, 'preference_learning'))
        os.mkdir(os.path.join(opt.demo_dir, 'preference_learning', opt.env_name + spec_kwargs))

    expert_traj = np.stack(expert_traj)
    if pl_model_file is not None:
        np.save(os.path.join(opt.demo_dir, 'preference_learning/' + filename + "_expert_traj.npy"), expert_traj)
    else:
        np.save(os.path.join(opt.demo_dir, filename + "_expert_traj.npy"), expert_traj)

    if len(expert_traj_extra) > 0 and pl_model_file is not None:
        expert_traj_extra = np.stack(expert_traj_extra)
        np.save(os.path.join(opt.demo_dir, filename + "_expert_traj_extra.npy"), expert_traj_extra)


def save_ranked_expert_demos(opt, model_dir, env_spec):
    sb_yml = open(opt.sb_config)
    sb_args = yaml.load(sb_yml)[opt.env_name]

    env_name = opt.env_name
    policy = sb_args['policy']

    # def without_keys(d, *keys):
        # return dict(filter(lambda key_value: key_value[0] not in keys, d.items()))

    # sb_args = without_keys(sb_args, 'n_envs', 'n_timesteps', 'policy',
                           # 'env_wrapper', 'normalize')

    sb_args = process_sb_args(sb_args)
    env, tenv = make_venv(opt, 1, env_spec, opt.env_spec_test, {})
    model = PPOSB(policy, env, **sb_args)

    # print(env_name, format_name_string(str(env_spec)))
    # model_file = [f for f in os.listdir(model_dir) if env_name + format_name_string(str(env_spec)) in f]

    # generate trajectories for every checkpoint in model_dir
    for model_file in os.listdir(model_dir):
        model = PPOSB.load(os.path.join(model_dir, model_file))
        save_expert_traj(opt, env, model, env_spec,
                            stable_baselines_model=True, pl_model_file=model_file)


def truncate_demos(demos, nr_traj):
    done_cnt = 0
    for i, t in enumerate(demos):
        if t[-1] == 1:
            done_cnt += 1
        if done_cnt >= nr_traj:
            return demos[:i]


def load_expert_demos(opt, env_name, env_spec=None):
    if opt.extra_demos:
        extra_string = '_expert_traj_extra.npy'
    else:
        extra_string = '_expert_traj.npy'
    try:
        expert_demos = {}
        if env_spec is None:
            demos = np.load(os.path.join(opt.demo_dir, env_name + extra_string))
            expert_demos['all'] = truncate_demos(demos, opt.num_expert_traj[0])
        else:
            for i, spec in enumerate(env_spec):
                demos = np.load(
                    os.path.join(opt.demo_dir, env_name + format_name_string(str(spec)) + extra_string))
                expert_demos[str(spec)] = truncate_demos(demos, opt.num_expert_traj[i])

    except:
        print("Train, generate and save expert trajectories using ppo algorithm first")
        assert False

    return expert_demos


def sample_trajectories(demos, nr_traj, length=10):
    traj = []
    for i in range(nr_traj):
        j = np.random.randint(0, len(demos))
        while j + length > len(demos):
            j = np.random.randint(0, len(demos))

        traj.append(demos[j:j + length])

    return np.stack(traj)


def prepare_update_gail(env, opt, expert_demos, obs, acs):
    ac_sample = env.action_space.sample()
    if isinstance(ac_sample, int):
        ac_shape = 1
    else:
        ac_shape = ac_sample.shape[-1]

    if isinstance(obs, torch.Tensor):
        obs = obs.cpu().numpy()
    if isinstance(acs, torch.Tensor):
        acs = acs.cpu().numpy()

    obs = np.reshape(obs, [-1, obs.shape[-1]])
    acs = np.reshape(acs, [-1, ac_shape])
    expert_ob_ac_done_reward = expert_demos[np.random.randint(0, expert_demos.shape[0], opt.n_steps * opt.n_envs), :]
    expert_ob_ac = expert_ob_ac_done_reward[:, :-2]
    expert_obs = expert_ob_ac[:, :-ac_shape]
    expert_acs = expert_ob_ac[:, -ac_shape:]
    policy_ob_ac = np.concatenate([obs, acs], 1)
    all_obs_ac = np.concatenate([expert_ob_ac, policy_ob_ac], axis=0)
    all_obs = np.concatenate([expert_obs, obs], axis=0)
    all_acs = np.concatenate([expert_acs, acs], axis=0)

    all_obs_t = torch.from_numpy(all_obs).type(torch.get_default_dtype())
    all_acs_t = torch.from_numpy(all_acs).type(torch.get_default_dtype())

    if torch.cuda.is_available():
        all_obs_t = all_obs_t.cuda()
        all_acs_t = all_acs_t.cuda()

    update_dict = {}
    update_dict['all_obs'] = all_obs_t
    update_dict['all_acs'] = all_acs_t

    return update_dict

def prepare_update_swil(env, opt, expert_demos, obs, acs):
    ac_sample = env.action_space.sample()
    if isinstance(ac_sample, int):
        ac_shape = 1
    else:
        ac_shape = ac_sample.shape[-1]

    if isinstance(obs, torch.Tensor):
        obs = obs.cpu().numpy()
    if isinstance(acs, torch.Tensor):
        acs = acs.cpu().numpy()

    obs = np.reshape(obs, [-1, obs.shape[-1]])
    acs = np.reshape(acs, [-1, ac_shape])
    expert_ob_ac_done_reward = expert_demos[np.random.randint(0, expert_demos.shape[0], opt.n_steps * opt.n_envs), :]
    expert_ob_ac = expert_ob_ac_done_reward[:, :-2]
    expert_obs = expert_ob_ac[:, :-ac_shape]
    expert_acs = expert_ob_ac[:, -ac_shape:]
    policy_ob_ac = np.concatenate([obs, acs], 1)
    all_obs_ac = np.concatenate([expert_ob_ac, policy_ob_ac], axis=0)
    all_obs = np.concatenate([expert_obs, obs], axis=0)
    all_acs = np.concatenate([expert_acs, acs], axis=0)

    all_obs_t = torch.from_numpy(all_obs).type(torch.get_default_dtype())
    all_acs_t = torch.from_numpy(all_acs).type(torch.get_default_dtype())

    if torch.cuda.is_available():
        all_obs_t = all_obs_t.cuda()
        all_acs_t = all_acs_t.cuda()

    update_dict = {}
    update_dict['all_obs'] = all_obs_t
    update_dict['all_acs'] = all_acs_t

    return update_dict

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
    expert_ob_ac_done_reward = expert_demos[np.random.randint(0, expert_demos.shape[0], opt.batch_size), :]
    expert_dones = expert_ob_ac_done_reward[:, -1]
    rewards = expert_ob_ac_done_reward[:, -2]
    expert_ob_ac = expert_ob_ac_done_reward[:, :-2]
    expert_obs = expert_ob_ac[:, :-ac_shape]
    expert_acs = expert_ob_ac[:, -ac_shape:]
    expert_obs_next = np.concatenate([expert_obs[1:], np.expand_dims(expert_obs[-1], 0)],
                                     axis=0)  # repeat last observation

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
    lprobs = np.concatenate([expert_lprobs_t.cpu().numpy(), policy_lprobs_t.cpu().numpy()], axis=0)

    expert_obs_t = torch.from_numpy(expert_obs).type(torch.get_default_dtype())
    expert_acs_t = torch.from_numpy(expert_acs).type(torch.get_default_dtype())
    expert_obs_next_t = torch.from_numpy(expert_obs_next).type(torch.get_default_dtype())
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
    update_dict['expert_obs'] = expert_obs_t
    update_dict['expert_obs_next'] = expert_obs_next_t
    update_dict['expert_acs'] = expert_acs_t
    update_dict['expert_lprobs'] = expert_lprobs_t
    update_dict['expert_dones'] = expert_dones_t

    update_dict['policy_obs'] = policy_obs_t
    update_dict['policy_obs_next'] = policy_obs_next_t
    update_dict['policy_acs'] = policy_acs_t
    update_dict['policy_lprobs'] = policy_lprobs_t

    update_dict['all_obs'] = all_obs_t
    update_dict['all_obs_next'] = all_obs_next_t
    update_dict['all_acs'] = all_acs_t
    update_dict['all_lprobs'] = all_lprobs_t

    return update_dict


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func


def process_sb_args(sb_args):

    def without_keys(d, *keys):
        return dict(filter(lambda key_value: key_value[0] not in keys, d.items()))

    sb_args = without_keys(sb_args, 'n_envs', 'n_timesteps', 'policy',
                           'env_wrapper', 'normalize')

    # # process policy_kwargs str
    if 'policy_kwargs' in sb_args.keys():
        if isinstance(sb_args['policy_kwargs'], str):
            sb_args['policy_kwargs'] = eval(sb_args['policy_kwargs'])

    # process schedules
    for key in ["learning_rate", "clip_range", "clip_range_vf"]:
        if key not in sb_args:
            continue
        if isinstance(sb_args[key], str):
            schedule, initial_value = sb_args[key].split("_")
            initial_value = float(initial_value)
            sb_args[key] = linear_schedule(initial_value)

    return sb_args


def train_using_sb(opt, save_checkpoints_for_pl=None,
                   pref_reward_model=None, irl_reward_model=None):
    '''
        :param opt: dict (NamedParameters) of argparse options
        :param save_checkpoints_for_pl: save intermediate model checkpoint files for preference learning
        :param pref_reward_model: load pretrained preference reward model
        :param irl_reward_model: load pretrained discriminator
        :return: None
    '''
    sb_yml = open(opt.sb_config)
    sb_args = yaml.load(sb_yml)[opt.env_name]
    set_random_seed(opt.seed)

    # multiprocess environment
    env_name = opt.env_name
    n_envs = sb_args['n_envs']
    n_timesteps = int(sb_args['n_timesteps'])
    policy = sb_args['policy']

    wrapper_kwargs = {}
    if pref_reward_model is not None:
        from preference import PreferenceReward
        pref_reward = PreferenceReward(opt)
        env = gym.make(env_name)
        init_obs = env.reset()
        print('Loading preference reward: ', pref_reward_model, ' use_actions: ', opt.use_actions)
        pref_reward.reward.load_state_dict(torch.load(pref_reward_model))
        wrapper_kwargs = {'init_obs': init_obs, 'reward_fn': pref_reward.reward, 'use_actions': opt.use_actions}

    if irl_reward_model is not None:
        use_cnn_base = False
        if 'MiniGrid' in opt.env_name:
            nr_actions = 7
            if opt.minigrid_wrapper == 'img':
                ac_base_type = 'minigridcnn'
                use_cnn_base = True

        import json
        from argparse import Namespace
        opt_test = json.load(open(os.path.join(os.path.dirname(irl_reward_model), 'args.json')))
        opt_test = Namespace(**opt_test)
        if opt.discriminator_type == 'airl':
            from irl import AIRLDiscriminator
            disc_test = AIRLDiscriminator(gym.make(opt_test.env_name, **opt.env_spec_test),
                                          opt_test.d_layer_dims,
                                          lr=opt_test.lr,
                                          gamma=opt_test.gamma,
                                          use_actions=opt_test.use_actions,
                                          irm_coeff=opt_test.irm_coeff,
                                          use_cnn_base=False)
            disc_test.load_state_dict(torch.load(irl_reward_model))
            wrapper_kwargs = {'disc': disc_test}

        elif opt.discriminator_type == 'gail':
            from irl import GAILDiscriminator

            disc_test = GAILDiscriminator(gym.make(opt_test.env_name, **opt.env_spec_test),
                                          opt_test.d_layer_dims,
                                          lr=opt_test.lr,
                                          gamma=opt_test.gamma,
                                          use_actions=opt_test.use_actions,
                                          irm_coeff=opt_test.irm_coeff,
                                          use_cnn_base=False)
            disc_test.load_state_dict(torch.load(irl_reward_model))
            wrapper_kwargs = {'disc': disc_test}

        if opt.discriminator_type == 'meirl':
            from irl import MEIRLDiscriminator
            disc_test = MEIRLDiscriminator(gym.make(opt_test.env_name, **opt.env_spec_test),
                                          opt_test.d_layer_dims,
                                          lr=opt_test.lr,
                                          use_actions=opt_test.use_actions,
                                          irm_coeff=opt_test.irm_coeff,
                                          use_cnn_base=False)
            disc_test.load_state_dict(torch.load(irl_reward_model))
            wrapper_kwargs = {'disc': disc_test}


    # train a policy for every modified env specified according to spec_kwargs (e.g. varying gravity)
    if opt.env_kwargs is None:
        list_kwargs = [None]
    else:
        list_kwargs = opt.env_kwargs

    for spec_kwargs in list_kwargs:
        if spec_kwargs is None:
            spec_kwargs = {}

        env, test_env = make_venv(opt, opt.n_envs, spec_kwargs, opt.env_spec_test, wrapper_kwargs)
        sb_args = process_sb_args(sb_args)

        if not os.path.exists('./sb_models'):
            os.mkdir('sb_models')
        if not os.path.exists('./sb_models/ckpt'):
            os.mkdir('sb_models/ckpt')

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(opt.output_dir, 'ppo_sb_' + opt.env_name + '_'
                                                 + str(spec_kwargs) + '_seed_' + str(opt.seed) + '_' + ts + '_' + opt.exp_id)
        sw = SummaryWriter(log_path)
        sw.add_text('params', str(opt))

        model = PPOSB(policy, env, **sb_args, tensorboard_log=log_path)
        new_logger = configure_logger(tensorboard_log=log_path)
        model.set_logger(new_logger)
        model_filename = "sb_models/ppo2_" + env_name + format_name_string(str(spec_kwargs))

        if not opt.load_sb_model:
            if save_checkpoints_for_pl is not None:
                checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=save_checkpoints_for_pl,
                                                         name_prefix='ppo_model')
                print("-" * 100)
                print(">>> SB Training for preference learning using checkpoint callback")
                print("-" * 100)
                model.learn(total_timesteps=n_timesteps, callback=checkpoint_callback)
            elif opt.pretrained_pref_reward is not None or opt.pretrained_irl_reward is not None:
                gt_reward_callback = CustomEpRewardCallback(log_path)
                eval_callback = EvalCallback(test_env, best_model_save_path=log_path,
                                                            log_path=log_path, eval_freq=10000,
                                                            deterministic=True, render=False)
                callback_list = CallbackList([gt_reward_callback, eval_callback])
                print("-" * 100)
                print(">>> SB Training using custom reward")
                print("-" * 100)
                model.learn(total_timesteps=n_timesteps, callback=callback_list)
            else:
                eval_callback = EvalCallback(test_env, best_model_save_path=log_path,
                                                            log_path=log_path, eval_freq=10000,
                                                            deterministic=True, render=False)
                print("-" * 100)
                print(">>> SB Training using ground truth reward")
                print("-" * 100)
                model.learn(total_timesteps=n_timesteps, callback=eval_callback)

            model.save(model_filename)
        else:
            print(">>> Loading model from checkpoint: " + model_filename)
            model = PPOSB.load(model_filename)

        # saving checkpoints means we want the ranked demonstrations:
        if save_checkpoints_for_pl is not None:
            save_ranked_expert_demos(opt, save_checkpoints_for_pl, spec_kwargs)
        else:
            if 'MiniGrid' in env_name:
                # eval_env = make_minigrid_venv(env_name, env_kwargs=spec_kwargs, n_envs=1,
                # wrapper_kwargs={}, seed=opt.seed)
                #_env = gym.make(env_name, **spec_kwargs)
                #eval_env = CustomImgObsWrapper(_env)
                _, eval_env = make_venv(opt,
                                        n_envs=1,
                                        spec=spec_kwargs,
                                        spec_test=spec_kwargs, wrapper_kwargs={})

            elif 'robosuite' in env_name:
                env_name = env_name.replace('robosuite-', '')
                eval_env = make_robosuite_env(env_name, spec_kwargs)
            else:
                eval_env = gym.make(env_name, **spec_kwargs)
                if 'pointMass' in env_name:
                    eval_env = pmObsWrapper(eval_env)

            save_expert_traj(opt, eval_env, model,
                             spec_kwargs, extra_reward_threshold=opt.threshold_reward,
                             nr_trajectories=opt.num_expert_traj, stable_baselines_model=True)

##########################################################################################################################
#### VamPPrior stuff
##########################################################################################################################

def log_Normal_diag(x, mean, log_var, average=False, dim=None):
    log_normal = -0.5 * ( log_var + torch.pow( x - mean, 2 ) / torch.exp( log_var ) )
    if average:
        return torch.mean( log_normal, dim )
    else:
        return torch.sum( log_normal, dim )

def log_Normal_standard(x, average=False, dim=None):
    log_normal = -0.5 * torch.pow( x , 2 )
    if average:
        return torch.mean( log_normal, dim )
    else:
        return torch.sum( log_normal, dim )

def vampprior_kld_vae(sample_dict, n_pseudo_inputs):
    mu = sample_dict['z1_mu']
    logvar = sample_dict['z1_logvar']
    z = sample_dict['z1']
    z_p_mu = sample_dict['z1_p_mu']
    z_p_logvar = sample_dict['z1_p_logvar']

    z_expanded = z.unsqueeze(1)
    means = z_p_mu.unsqueeze(0)
    logvars = z_p_logvar.unsqueeze(0)
    log_normal = log_Normal_diag(z_expanded, means, logvars, dim=2) - np.log(n_pseudo_inputs)
    #if args.attention and args.attention_type == 'nn':
    #    log_normal = vdb.attention_weights * log_normal
    log_normal_max, _ = torch.max(log_normal, 1)
    log_p_z = log_normal_max + torch.log(torch.sum(torch.exp(log_normal - log_normal_max.unsqueeze(1)), 1))
    log_q_z = log_Normal_diag(z, mu, logvar, dim=1)
    kld = -(log_p_z - log_q_z)
    kld = kld.mean()
    return kld

def vampprior_kld_twolayervae(sample_dict, n_pseudo_inputs, use_vampprior=True):
    log_p_z1 = log_Normal_diag(sample_dict['z1'], sample_dict['z1_p_mu'], sample_dict['z1_p_logvar'], dim=1)
    log_q_z1 = log_Normal_diag(sample_dict['z1'], sample_dict['z1_mu'], sample_dict['z1_logvar'], dim=1)
    log_q_z2 = log_Normal_diag(sample_dict['z2'], sample_dict['z2_mu'], sample_dict['z2_logvar'], dim=1)
    if use_vampprior:
        z_expanded = sample_dict['z2'].unsqueeze(1)
        means = sample_dict['z2_p_mu'].unsqueeze(0)
        logvars = sample_dict['z2_p_logvar'].unsqueeze(0)
        log_normal = log_Normal_diag(z_expanded, means, logvars, dim=2) - np.log(n_pseudo_inputs)
        log_normal_max, _ = torch.max(log_normal, 1)
        log_p_z2 = log_normal_max + torch.log(torch.sum(torch.exp(log_normal - log_normal_max.unsqueeze(1)), 1))
    else:
        log_p_z2 = log_Normal_standard(sample_dict['z2'], dim=1)
    kld = -(log_p_z1 + log_p_z2 - log_q_z1 - log_q_z2)
    kld = kld.mean()
    return kld

def gaussian_kld(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1)


class CustomEpRewardCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, log_path, verbose=0):
        super(CustomEpRewardCallback, self).__init__(verbose)

        self.cnt = 0
        self.aux_writer = SummaryWriter(log_path)
        self.episode_reward_gt = 0
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
        self.episode_reward_gt = 0
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        self.cnt += 1
        # get info
        infos = self.locals['infos']
        rew = np.mean([inf['gt_reward'] for inf in infos])
        self.episode_reward_gt += rew
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.aux_writer.add_scalar('Reward/Ep_rewards_gt', self.episode_reward_gt, self.cnt)


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
        #self.n_calls = 0  # type: int
        #self.num_timesteps = 0  # type: int
        # local and global variables
        #self.locals = None  # type: Dict[str, Any]
        #self.globals = None  # type: Dict[str, Any]
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
        rews = self.locals['rewards']
        infos = self.locals['infos']
        rew = np.mean(rews)
        rew_gt = np.mean([inf['gt_reward'] if 'gt_reward' in inf.keys() else 0 for inf in infos])
        self.episode_reward += rew
        self.episode_reward_gt += rew_gt
        self.cnt += 1

        return self._on_step()

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.aux_writer.add_scalar('Reward/Ep_rewards_' + self.id, self.episode_reward, self.cnt)
        self.aux_writer.add_scalar('Reward/Ep_rewards_gt_' + self.id, self.episode_reward_gt, self.cnt)

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

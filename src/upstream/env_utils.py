import numpy as np
import re
import glob
import torch
import itertools
from itertools import count
from datetime import datetime
from collections import deque
import os

# import doorenv
# import envs
import gym
from gym import Env
from gym.spaces import Box
from gym.spaces import Discrete
import pybulletgym
import pointMass
import dm_control

# import dmc2gym
# import gym_minigrid
# import robosuite
# from robosuite.wrappers import GymWrapper
# from robosuite import load_controller_config
# from gym_minigrid.wrappers import *
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# from stable_baselines3.bench import Monitor
# from stable_baselines3.common.logger import Logger
# from stable_baselines3.common.monitor import Monitor
import copy

import dm_env
import ot
from sklearn import preprocessing


def get_trajectory_list(demos):
    done_cnt = 0
    trajs = []
    ep = []
    for i, t in enumerate(demos):
        ep.append(t)
        if t[-1] == 1:
            done_cnt += 1
            trajs.append(np.array(ep))
            ep = []

    return trajs


def format_name_string(name_string):
    name_string = (
        name_string.replace("{", "_")
        .replace("}", "")
        .replace(" ", "")
        .replace("'xml_file'", "")
    )
    name_string = name_string.replace("'", "").replace(":", "").replace("/", "")

    return name_string


def get_env_demo_files(expert_demo_dir, env_name, spec):
    demo_dir = os.listdir(expert_demo_dir)
    if spec is not None:
        specd_env_name = env_name + format_name_string(str(spec))
    else:
        specd_env_name = env_name

    demo_files = [f for f in demo_dir if specd_env_name in f]

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split(r"(\d+)", text)]

    demo_files.sort(key=natural_keys)

    return demo_files


def make_venv(
    opt, n_envs, spec, spec_test, wrapper_kwargs, use_rank=True, use_subprocess=False
):
    if spec is None:
        spec = {}
    if spec_test is None:
        spec_test = {}
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    print("-" * 100)

    if "Custom" in opt.env_name:
        envs = make_pybullet_venv(
            opt.env_name,
            env_kwargs=spec,
            wrapper_kwargs=wrapper_kwargs,
            n_envs=n_envs,
            seed=opt.seed,
            use_subprocess=use_subprocess,
            use_rank=use_rank,
        )
        testing_env = make_pybullet_venv(
            opt.env_name,
            env_kwargs=spec_test,
            n_envs=1,
            seed=opt.seed,
            use_subprocess=use_subprocess,
            use_rank=use_rank,
        )
    elif "MiniGrid" in opt.env_name:
        envs = make_minigrid_venv(
            opt.env_name,
            env_kwargs=spec,
            wrapper_kwargs=wrapper_kwargs,
            n_envs=n_envs,
            seed=opt.seed,
            wrapper_type=opt.minigrid_wrapper,
            use_subprocess=use_subprocess,
            use_rank=use_rank,
        )
        testing_env = make_minigrid_venv(
            opt.env_name,
            env_kwargs=spec_test,
            n_envs=1,
            seed=opt.seed,
            use_subprocess=use_subprocess,
            use_rank=use_rank,
            wrapper_type=opt.minigrid_wrapper,
        )
    elif "robosuite" in opt.env_name:
        env_name = opt.env_name.replace("robosuite-", "")
        envs = make_robosuite_venv(
            env_name,
            env_kwargs=spec,
            wrapper_kwargs=wrapper_kwargs,
            n_envs=n_envs,
            seed=opt.seed,
            use_subprocess=use_subprocess,
            use_rank=use_rank,
        )
        testing_env = make_robosuite_venv(
            env_name,
            env_kwargs=spec_test,
            n_envs=1,
            seed=opt.seed,
            use_subprocess=use_subprocess,
            use_rank=use_rank,
        )

    # dm_control environments
    elif "dmc" in opt.env_name:
        _, env_name, env_task = opt.env_name.split("-")
        envs = make_dmc_venv(
            env_name,
            env_task,
            env_kwargs=spec,
            wrapper_kwargs=wrapper_kwargs,
            n_envs=n_envs,
            seed=opt.seed,
            use_subprocess=use_subprocess,
            use_rank=use_rank,
        )
        testing_env = make_dmc_venv(
            env_name,
            env_task,
            env_kwargs=spec_test,
            n_envs=1,
            seed=opt.seed,
            use_subprocess=use_subprocess,
            use_rank=use_rank,
        )

    elif "door" in opt.env_name:
        envs = make_door_venv(
            opt.env_name,
            env_kwargs=spec,
            wrapper_kwargs=wrapper_kwargs,
            n_envs=n_envs,
            seed=opt.seed,
            use_subprocess=use_subprocess,
            use_rank=use_rank,
        )
        testing_env = make_door_venv(
            opt.env_name,
            env_kwargs=spec_test,
            n_envs=1,
            seed=opt.seed,
            use_subprocess=use_subprocess,
            use_rank=use_rank,
        )

    elif "pointMass" in opt.env_name:
        envs = make_pointmass_venv(
            opt.env_name,
            env_kwargs=spec,
            wrapper_kwargs=wrapper_kwargs,
            n_envs=n_envs,
            seed=opt.seed,
            use_subprocess=use_subprocess,
            use_rank=use_rank,
        )
        testing_env = make_pointmass_venv(
            opt.env_name,
            env_kwargs=spec_test,
            n_envs=1,
            seed=opt.seed,
            use_subprocess=use_subprocess,
            use_rank=use_rank,
        )
        testing_env = gym.make(opt.env_name, **spec_test)
        testing_env = pmObsWrapper(testing_env)

    # not pybullet or minigrid
    else:

        def make_env(rank):
            def _thunk():
                env = gym.make(opt.env_name, **spec)
                if use_rank:
                    seed = opt.seed + rank
                else:
                    seed = opt.seed
                env.seed(seed)
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                env = apply_wrappers(env, **wrapper_kwargs)
                return env

            return _thunk

        envs = [make_env(i) for i in range(n_envs)]
        if use_subprocess:
            envs = SubprocVecEnv(envs)
        else:
            envs = DummyVecEnv(envs)
        testing_env = gym.make(opt.env_name, **spec_test)
        testing_env.seed(opt.seed)
        testing_env.action_space.seed(opt.seed)
        testing_env.observation_space.seed(opt.seed)

    print("-" * 100)

    return envs, testing_env


# The point of this is to include custom wrappers before creating vectorized env
# This is adapted from stable baselines
def make_pybullet_venv(
    env_id,
    n_envs,
    seed,
    env_kwargs=None,
    wrapper_kwargs=None,
    allow_early_resets=True,
    start_method=None,
    use_rank=True,
    use_subprocess=False,
):
    """
    Create a wrapped, monitored VecEnv for Mujoco.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param wrapper_kwargs: (dict) the parameters for wrap_deepmind function
    :param start_index: (int) start rank index
    :param allow_early_resets: (bool) allows early reset of the environment
    :param start_method: (str) method used to start the subprocesses.
        See SubprocVecEnv doc for more information
    :param use_subprocess: (bool) Whether to use `SubprocVecEnv` or `DummyVecEnv` when
        `num_env` > 1, `DummyVecEnv` is usually faster. Default: False
    :return: (VecEnv) The atari environment
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    print(">>> Making environments with parameters: ", env_kwargs)

    def make_env(rank):
        def _thunk():
            env = gym.make(env_id, **env_kwargs)
            if use_rank:
                seedr = seed + rank
            else:
                seedr = seed
            env.seed(seedr)
            env.action_space.seed(seedr)
            env.observation_space.seed(seedr)
            # env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
            # allow_early_resets=allow_early_resets)
            return apply_wrappers(env, **wrapper_kwargs)
            # return env

        return _thunk

    # When using one environment, no need to start subprocesses
    if n_envs == 1 or not use_subprocess:
        return DummyVecEnv([make_env(i) for i in range(n_envs)])

    return SubprocVecEnv(
        [make_env(i) for i in range(n_envs)], start_method=start_method
    )


def make_dmc_venv(
    env_id,
    env_task,
    n_envs,
    seed,
    env_kwargs=None,
    wrapper_kwargs=None,
    allow_early_resets=True,
    start_method=None,
    use_rank=True,
    use_subprocess=False,
):
    """ """

    def make_env(rank):
        def _thunk():
            if use_rank:
                seedr = seed + rank
            else:
                seedr = seed
            env = dmc2gym.make(env_id, env_task, seed=seedr)
            # TODO: modify env according to env_kwargs
            env.action_space.seed(seedr)
            env.observation_space.seed(seedr)
            # env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
            # allow_early_resets=allow_early_resets)
            return apply_wrappers(env, **wrapper_kwargs)
            # return env

        return _thunk

    # When using one environment, no need to start subprocesses
    if n_envs == 1 or not use_subprocess:
        return DummyVecEnv([make_env(i) for i in range(n_envs)])

    return SubprocVecEnv(
        [make_env(i) for i in range(n_envs)], start_method=start_method
    )


def make_robosuite_env(env_name, env_kwargs):
    # load OSC controller to use for all environments
    controller = load_controller_config(default_controller="OSC_POSE")

    # these arguments are the same for all envs
    config = {
        "controller_configs": controller,
        "horizon": 500,
        "gripper_types": "default",
        "control_freq": 40,
        "reward_shaping": True,
        "has_renderer": False,
        "reward_scale": 1.0,
        "use_camera_obs": False,
        "use_object_obs": True,
        "ignore_done": False,
        "hard_reset": False,
    }

    # this should be used during training to speed up training
    # A renderer should be used if you're visualizing rollouts!
    config["has_offscreen_renderer"] = False

    env = robosuite.make(
        env_name=env_name,  # try with other tasks like "Stack" and "Door"
        # try with other robots like "Sawyer" and "Jaco"
        robots=list(env_kwargs.values())[0],
        **config
    )

    return NormalizedBoxEnv(GymWrapper(env))


# The point of this is to include custom wrappers before creating vectorized env
# This is adapted from stable baselines


def make_robosuite_venv(
    env_id,
    n_envs,
    seed,
    env_kwargs=None,
    wrapper_kwargs=None,
    allow_early_resets=True,
    start_method=None,
    use_rank=True,
    use_subprocess=False,
):
    """
    Create a wrapped, monitored VecEnv for Mujoco.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param wrapper_kwargs: (dict) the parameters for wrap_deepmind function
    :param start_index: (int) start rank index
    :param allow_early_resets: (bool) allows early reset of the environment
    :param start_method: (str) method used to start the subprocesses.
        See SubprocVecEnv doc for more information
    :param use_subprocess: (bool) Whether to use `SubprocVecEnv` or `DummyVecEnv` when
        `num_env` > 1, `DummyVecEnv` is usually faster. Default: False
    :return: (VecEnv) The atari environment
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    print(">>> Making environments with parameters: ", env_kwargs)

    def make_env(rank):
        def _thunk():
            # create environment instance
            env = make_robosuite_env(env_id, env_kwargs)
            # env = gym.make(env_id, **env_kwargs)
            if use_rank:
                seedr = seed + rank
            else:
                seedr = seed
            env.seed(seedr)
            env.action_space.seed(seedr)
            env.observation_space.seed(seedr)
            # env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
            # allow_early_resets=allow_early_resets)
            return apply_wrappers(env, **wrapper_kwargs)
            # return env

        return _thunk

    # When using one environment, no need to start subprocesses
    if n_envs == 1 or not use_subprocess:
        return DummyVecEnv([make_env(i) for i in range(n_envs)])

    return SubprocVecEnv(
        [make_env(i) for i in range(n_envs)], start_method=start_method
    )


def make_door_venv(
    env_id,
    n_envs,
    seed,
    env_kwargs=None,
    wrapper_kwargs=None,
    start_method=None,
    use_rank=True,
    use_subprocess=False,
):
    """
    Create a wrapped, monitored VecEnv for Mujoco.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param wrapper_kwargs: (dict) the parameters for wrap_deepmind function
    :param start_index: (int) start rank index
    :param allow_early_resets: (bool) allows early reset of the environment
    :param start_method: (str) method used to start the subprocesses.
        See SubprocVecEnv doc for more information
    :param use_subprocess: (bool) Whether to use `SubprocVecEnv` or `DummyVecEnv` when
        `num_env` > 1, `DummyVecEnv` is usually faster. Default: False
    :return: (VecEnv) The atari environment
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    print(">>> Making environments with parameters: ", env_kwargs)

    def make_env(rank):
        def _thunk():
            # create environment instance
            env = gym.make(env_id, **env_kwargs)
            if use_rank:
                seedr = seed + rank
            else:
                seedr = seed
            env.seed(seedr)
            env.action_space.seed(seedr)
            env.observation_space.seed(seedr)
            # env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
            # allow_early_resets=allow_early_resets)
            return apply_wrappers(env, **wrapper_kwargs)
            # return env

        return _thunk

    # When using one environment, no need to start subprocesses
    if n_envs == 1 or not use_subprocess:
        return DummyVecEnv([make_env(i) for i in range(n_envs)])

    return SubprocVecEnv(
        [make_env(i) for i in range(n_envs)], start_method=start_method
    )


def make_minigrid_venv(
    env_id,
    n_envs,
    seed,
    env_kwargs=None,
    wrapper_type="flat",
    wrapper_kwargs=None,
    allow_early_resets=True,
    start_method=None,
    use_rank=True,
    use_subprocess=False,
):
    """
    Create a wrapped, monitored VecEnv for Minigrid.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param wrapper_kwargs: (dict) the parameters for wrap_deepmind function
    :param start_index: (int) start rank index
    :param allow_early_resets: (bool) allows early reset of the environment
    :param start_method: (str) method used to start the subprocesses.
        See SubprocVecEnv doc for more information
    :param use_subprocess: (bool) Whether to use `SubprocVecEnv` or `DummyVecEnv` when
        `num_env` > 1, `DummyVecEnv` is usually faster. Default: False
    :return: (VecEnv) The atari environment
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    print(">>> Making environments with parameters: ", env_kwargs)

    def make_env(rank):
        def _thunk():
            env = gym.make(env_id, **env_kwargs)
            if use_rank:
                seedr = seed + rank
            else:
                seedr = seed
            env.seed(seedr)
            env.action_space.seed(seedr)
            env.observation_space.seed(seedr)
            # env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
            # allow_early_resets=allow_early_resets)
            env = apply_wrappers(env, **wrapper_kwargs)
            if wrapper_type == "flat":
                return FlatObsWrapper(env, **wrapper_kwargs)
            else:
                return ImgObsWrapper(env, **wrapper_kwargs)

        return _thunk

    if n_envs == 1 or not use_subprocess:
        return DummyVecEnv([make_env(i) for i in range(n_envs)])

    return SubprocVecEnv(
        [make_env(i) for i in range(n_envs)], start_method=start_method
    )


def make_pointmass_venv(
    env_id,
    n_envs,
    seed,
    env_kwargs=None,
    wrapper_kwargs=None,
    allow_early_resets=True,
    start_method=None,
    use_rank=True,
    use_subprocess=False,
):
    """
    Create a wrapped, monitored VecEnv for pointmass.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param wrapper_kwargs: (dict) the parameters for wrap_deepmind function
    :param start_index: (int) start rank index
    :param allow_early_resets: (bool) allows early reset of the environment
    :param start_method: (str) method used to start the subprocesses.
        See SubprocVecEnv doc for more information
    :param use_subprocess: (bool) Whether to use `SubprocVecEnv` or `DummyVecEnv` when
        `num_env` > 1, `DummyVecEnv` is usually faster. Default: False
    :return: (VecEnv) The atari environment
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    print(">>> Making environments with parameters: ", env_kwargs)

    def make_env(rank):
        def _thunk():
            env = gym.make(env_id, **env_kwargs)
            if use_rank:
                seedr = seed + rank
            else:
                seedr = seed
            env.seed(seedr)
            env.action_space.seed(seedr)
            env.observation_space.seed(seedr)
            # env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
            # allow_early_resets=allow_early_resets)
            env = apply_wrappers(env, **wrapper_kwargs)
            return pmObsWrapper(env, **wrapper_kwargs)

        return _thunk

    if n_envs == 1 or not use_subprocess:
        return DummyVecEnv([make_env(i) for i in range(n_envs)])

    return SubprocVecEnv(
        [make_env(i) for i in range(n_envs)], start_method=start_method
    )


def apply_wrappers(env, reward_fn=None, stack_size=1, disc=None, demos=None, **kwargs):
    if stack_size > 1:
        env = FeatureStack(env, stack_size)
    if reward_fn is not None:
        env = CustomReward(
            env,
            reward_fn,
            init_obs=kwargs["init_obs"],
            use_actions=kwargs["use_actions"],
        )
    if disc is not None:
        if "use_actions" in kwargs.keys():
            env = DiscReward(env, disc, kwargs["use_actions"])
        else:
            env = DiscReward(env, disc)

    if demos is not None:
        env = PWILReward(env, demos, use_actions=kwargs["use_actions"])

    return env


def repack_vecenv(vecenv, disc, use_subprocess=False):
    def repack_env(e):
        def _thunk():
            env = apply_wrappers(e, disc=disc)
            return env

        return _thunk

    env_list = [repack_env(env) for env in vecenv.envs]
    if use_subprocess:
        return SubprocVecEnv(env_list)
    else:
        return DummyVecEnv(env_list)


# Gym wrapper classes
class FeatureStack(gym.Wrapper):
    def __init__(self, env, stack_size=2):
        super().__init__(env=env)

        self.stack_size = stack_size
        self.stack = deque([], self.stack_size)

    def reset(self):
        obs = self.env.reset()

        first_features = self.env.get_current_features()
        for _ in range(self.stack_size):
            self.stack.append(first_features)

        return obs

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.stack.append(info["features"])

        info["features"] = np.concatenate(list(self.stack), axis=0)
        # the key name "visual_features" is for compatibility with the
        # VisualHistoryWrapper
        # info["visual_features"] = LazyFeatures(list(self.stack))

        return state, reward, done, info


class CustomReward(gym.Wrapper):
    def __init__(self, env, reward_fn, init_obs=None, use_actions=False):
        super().__init__(env=env)
        # self.reward_fn = make_network(**reward_fn_spec)
        self.use_actions = use_actions
        self.reward_fn = reward_fn
        self.obs = None

    def step(self, action):
        # also, here, WANT state before applying action
        next_obs, gt_reward, done, info = self.env.step(action)
        if self.obs is None:
            obs = next_obs
        else:
            obs = self.obs
        info["gt_reward"] = gt_reward
        with torch.no_grad():
            # here either use observation or features from info dict
            # reward = self.reward_fn(torch.tensor(info["features"],
            # dtype=torch.float32))
            if self.use_actions:
                if not isinstance(action, np.ndarray):
                    action = [action]
                reward = self.reward_fn(
                    torch.cat(
                        [
                            torch.tensor(obs, dtype=torch.get_default_dtype()),
                            torch.tensor(action, dtype=torch.get_default_dtype()),
                        ],
                        axis=-1,
                    )
                )
            else:
                reward = self.reward_fn(
                    torch.tensor(obs, dtype=torch.get_default_dtype())
                )
            reward = reward.unsqueeze(0).cpu().numpy()

        self.obs = next_obs

        return next_obs, reward, done, info


class PWILRewarder(object):
    """Rewarder class to compute PWIL rewards."""

    def __init__(
        self,
        demonstrations,
        subsampling,
        env,
        num_demonstrations=1,
        time_horizon=1000.0,
        alpha=5.0,
        beta=5.0,
        observation_only=False,
    ):

        self.num_demonstrations = num_demonstrations
        self.time_horizon = time_horizon
        self.subsampling = subsampling

        # Observations and actions are flat.
        ob_shapes = list(env.observation_space.shape)
        ac_shapes = list(env.action_space.shape)
        if len(ac_shapes) == 0:
            dim_act = 1
        else:
            dim_act = ac_shapes[-1]
        dim_obs = ob_shapes[-1]

        self.reward_sigma = beta * time_horizon / np.sqrt(dim_act + dim_obs)
        self.reward_scale = alpha

        self.observation_only = observation_only
        self.demonstrations = self.filter_demonstrations(
            get_trajectory_list(demonstrations)
        )
        # self.vectorized_demonstrations = self.vectorize(self.demonstrations)
        if self.observation_only:
            demos = np.concatenate(self.demonstrations)
            self.vectorized_demonstrations = demos[:, :dim_obs]
        else:
            demos = np.concatenate(self.demonstrations)
            self.vectorized_demonstrations = demos[:, : dim_obs + dim_act]

        self.scaler = self.get_scaler()
        self.reset()

    def filter_demonstrations(self, demonstrations):
        filtered_demonstrations = []
        np.random.shuffle(demonstrations)
        for episode in demonstrations[: self.num_demonstrations]:
            # Random episode start.
            random_offset = np.random.randint(0, self.subsampling)
            # Subsampling.
            subsampled_episode = episode[random_offset :: self.subsampling]
            # Specify step types of demonstrations.
            # for transition in subsampled_episode:
            #     transition['step_type'] = dm_env.StepType.MID
            # subsampled_episode[0]['step_type'] = dm_env.StepType.FIRST
            # subsampled_episode[-1]['step_type'] = dm_env.StepType.LAST
            filtered_demonstrations.append(subsampled_episode)
        return filtered_demonstrations

    def get_scaler(self):
        """Defines a scaler to derive the standardized Euclidean distance."""
        scaler = preprocessing.StandardScaler()
        scaler.fit(self.vectorized_demonstrations)
        return scaler

    def reset(self):
        """Makes all expert transitions available and initialize weights."""
        self.expert_atoms = copy.deepcopy(
            self.scaler.transform(self.vectorized_demonstrations)
        )
        num_expert_atoms = len(self.expert_atoms)
        self.expert_weights = np.ones(num_expert_atoms) / (num_expert_atoms)

    def compute_reward(self, obs, action=None):
        """Computes reward as presented in Algorithm 1."""
        # Scale observation and action.
        if action is None:
            agent_atom = obs
        else:
            if not isinstance(action, int):
                action = np.array([action])
            agent_atom = np.concatenate([obs, action])

        agent_atom = np.expand_dims(agent_atom, axis=0)  # add dim for scaler
        agent_atom = self.scaler.transform(agent_atom)[0]

        # reset if list exhausted
        if len(self.expert_atoms) == 1:
            self.reset()

        cost = 0.0
        # As we match the expert's weights with the agent's weights, we might
        # raise an error due to float precision, we substract a small epsilon from
        # the agent's weights to prevent that.
        weight = 1.0 / self.time_horizon - 1e-6
        norms = np.linalg.norm(self.expert_atoms - agent_atom, axis=1)

        while weight > 0:
            # Get closest expert state action to agent's state action.
            argmin = norms.argmin()
            expert_weight = self.expert_weights[argmin]

            # Update cost and weights.
            if weight >= expert_weight:
                weight -= expert_weight
                cost += expert_weight * norms[argmin]
                self.expert_weights = np.delete(self.expert_weights, argmin, 0)
                self.expert_atoms = np.delete(self.expert_atoms, argmin, 0)
                norms = np.delete(norms, argmin, 0)
            else:
                cost += weight * norms[argmin]
                self.expert_weights[argmin] -= weight
                weight = 0

        reward = self.reward_scale * np.exp(-self.reward_sigma * cost)
        return reward.astype("float32")

    def compute_w2_dist_to_expert(self, trajectory):
        """Computes Wasserstein 2 distance to expert demonstrations."""
        self.reset()
        if self.observation_only:
            trajectory = [t["observation"] for t in trajectory]
        else:
            trajectory = [
                np.concatenate([t["observation"], t["action"]]) for t in trajectory
            ]

        trajectory = self.scaler.transform(trajectory)
        trajectory_weights = 1.0 / len(trajectory) * np.ones(len(trajectory))
        cost_matrix = ot.dist(trajectory, self.expert_atoms, metric="euclidean")
        w2_dist = ot.emd2(trajectory_weights, self.expert_weights, cost_matrix)
        return w2_dist


class PWILReward(gym.Wrapper):
    def __init__(self, env, demos, n_demos, subsampling=20, use_actions=False):
        super().__init__(env=env)
        # self.reward_fn = make_network(**reward_fn_spec)
        self.use_actions = use_actions
        self.demos = demos
        self.obs = None
        self.pwil = PWILRewarder(
            demos,
            subsampling=subsampling,
            env=env,
            num_demonstrations=n_demos,
            observation_only=(not use_actions),
        )

    def step(self, action):
        # also, here, WANT state before applying action
        next_obs, gt_reward, done, info = self.env.step(action)
        if self.obs is None:
            obs = next_obs
        else:
            obs = self.obs
        info["gt_reward"] = gt_reward
        if self.use_actions:
            reward = self.pwil.compute_reward(obs, action)
        else:
            reward = self.pwil.compute_reward(obs, action=None)

        self.obs = next_obs

        return next_obs, reward, done, info


class DiscReward(gym.Wrapper):
    def __init__(self, env, discriminator, use_actions=False):
        super().__init__(env=env)
        # self.reward_fn = make_network(**reward_fn_spec)
        self.use_actions = use_actions
        self.discriminator = discriminator
        self.obs = None

    def step(self, action):
        next_obs, gt_reward, done, info = self.env.step(action)
        info["gt_reward"] = gt_reward
        if self.obs is not None:
            obs_t = torch.tensor(self.obs, dtype=torch.get_default_dtype())
        else:
            obs_t = torch.tensor(next_obs, dtype=torch.get_default_dtype())

        acs_t = torch.tensor(action, dtype=torch.get_default_dtype())
        next_obs_t = torch.tensor(next_obs, dtype=torch.get_default_dtype())

        with torch.no_grad():
            # get discriminator reward and train on that
            # print([p.norm(2) for p in self.discriminator.parameters()])
            irl_reward = self.discriminator.get_reward(obs_t, acs_t).cpu().numpy()
            # irl_reward = self.discriminator.get_reward(obs_t, acs_t, next_obs_t).cpu().numpy()
        self.obs = next_obs

        return next_obs, irl_reward, done, info


# pointmass dict extraction wrapper


class pmObsWrapper(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        self.env.observation_space = self.env.observation_space["observation"]
        print(self.env.observation_space)
        super().__init__(env=self.env)

    def reset(self):
        o = self.env.reset()
        return np.concatenate([o["full_positional_state"], o["desired_goal"]])

    def step(self, action):
        o, r, d, i = self.env.step(action)
        # shaped reward
        r_s = -np.linalg.norm(o["desired_goal"] - o["full_positional_state"])
        o = np.concatenate([o["full_positional_state"], o["desired_goal"]])
        return o, r_s, d, i


class CustomImgObsWrapper(gym.core.ObservationWrapper):
    """
    Use the transposed image as the only observation output, no language/mission.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces["image"]

    def observation(self, obs):
        obs_t = np.transpose(obs["image"], (2, 1, 0))
        print(obs_t.shape)
        return obs_t


# ---------------------------------------------------------------------------
# took these from rlkit implementation to normalized robosuite envs
# ---------------------------------------------------------------------------


class ProxyEnv(Env):
    def __init__(self, wrapped_env):
        self._wrapped_env = wrapped_env
        self.action_space = self._wrapped_env.action_space
        self.observation_space = self._wrapped_env.observation_space

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()

    def __getattr__(self, attr):
        if attr == "_wrapped_env":
            raise AttributeError()
        return getattr(self._wrapped_env, attr)

    def __getstate__(self):
        """
        This is useful to override in case the wrapped env has some funky
        __getstate__ that doesn't play well with overriding __getattr__.
        The main problematic case is/was gym's EzPickle serialization scheme.
        :return:
        """
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.wrapped_env)


class NormalizedBoxEnv(ProxyEnv):
    """
    Normalize action to in [-1, 1].
    Optionally normalize observations and scale reward.
    """

    def __init__(
        self,
        env,
        reward_scale=1.0,
        obs_mean=None,
        obs_std=None,
    ):
        ProxyEnv.__init__(self, env)
        self._should_normalize = not (obs_mean is None and obs_std is None)
        if self._should_normalize:
            if obs_mean is None:
                obs_mean = np.zeros_like(env.observation_space.low)
            else:
                obs_mean = np.array(obs_mean)
            if obs_std is None:
                obs_std = np.ones_like(env.observation_space.low)
            else:
                obs_std = np.array(obs_std)
        self._reward_scale = reward_scale
        self._obs_mean = obs_mean
        self._obs_std = obs_std
        ub = np.ones(self._wrapped_env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)

    def estimate_obs_stats(self, obs_batch, override_values=False):
        if self._obs_mean is not None and not override_values:
            raise Exception(
                "Observation mean and std already set. To "
                "override, set override_values to True."
            )
        self._obs_mean = np.mean(obs_batch, axis=0)
        self._obs_std = np.std(obs_batch, axis=0)

    def _apply_normalize_obs(self, obs):
        return (obs - self._obs_mean) / (self._obs_std + 1e-8)

    def step(self, action):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.0) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step
        if self._should_normalize:
            next_obs = self._apply_normalize_obs(next_obs)
        return next_obs, reward * self._reward_scale, done, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env

# from stable_baselines3.bench import Monitor
# from stable_baselines3.common.logger import Logger
# from stable_baselines3.common.monitor import Monitor
import copy
import logging
from collections import deque

# import doorenv
# import envs
import gym
import numpy as np
import ot
import torch
from sklearn import preprocessing

# import dmc2gym
# import gym_minigrid
# import robosuite
# from robosuite.wrappers import GymWrapper
# from robosuite import load_controller_config
# from gym_minigrid.wrappers import *
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

logger = logging.getLogger(__name__)


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
        self._logger = logger

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
        print("Number of demonstrations: ", len(demonstrations))
        np.random.shuffle(demonstrations)
        for episode in demonstrations[: self.num_demonstrations]:
            # Random episode start.
            random_offset = np.random.randint(0, self.subsampling)
            print("Random offset: ", random_offset)
            # Subsampling.
            print("Full episode length", len(episode))
            subsampled_episode = episode[random_offset :: self.subsampling]
            print("Subsampled episode length", len(subsampled_episode))
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
            if not isinstance(action, np.ndarray):
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
        # reset the pwil because the atoms are exhausted
        self.pwil.reset()
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

import os

from gym import Env
from stable_baselines3 import PPO as PPOSB
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from src.ours.util.common.helper import RewardPlotter, TqdmCallback
from src.ours.util.common.param import PwilParam
from src.ours.util.common.train import Trainer
from src.ours.util.expert.trajectory.manager import TrajectoryManager
from src.upstream.env_utils import PWILReward
from src.upstream.utils import CustomCallback


class TrainerPwil(Trainer):
    def __init__(
        self,
        training_param: PwilParam,
        envs_and_identifier: tuple[tuple[Env, Env], str],
    ):
        super().__init__(training_param)

        self._model_dir = self._training_param.model_dir
        self._save_deterministic = False

        (
            self._env_raw,
            self._env_raw_testing,
        ), self._env_identifier = envs_and_identifier

    def train(
        self,
        demos,
        n_demos,
        subsampling,
        use_actions,
        n_timesteps,
        fname,
    ):
        env = PWILReward(
            env=self._env_raw,
            demos=demos,
            n_demos=n_demos,
            subsampling=subsampling,
            use_actions=use_actions,
        )

        plot = RewardPlotter.plot_reward(discriminator=None, env=env)

        model = PPOSB(
            "MlpPolicy",
            env,
            verbose=0,
            **self._training_param.kwargs_ppo,
            tensorboard_log=self._training_param.sb3_tblog_dir
        )

        eval_callback = EvalCallback(
            self._env_raw_testing,
            best_model_save_path=self._training_param.sb3_tblog_dir,
            log_path=self._training_param.sb3_tblog_dir,
            eval_freq=10000,
            deterministic=True,
            render=False,
        )

        # eval_callback.init_callback(ppo_dict[k])
        callback_list = CallbackList(
            [
                CustomCallback(id="", log_path=self._training_param.sb3_tblog_dir),
                eval_callback,
                TqdmCallback(),
            ]
        )

        model.learn(total_timesteps=n_timesteps, callback=callback_list)

        model.save(
            os.path.join(self._model_dir, "model_" + fname + str(int(n_timesteps)))
        )
        TrajectoryManager(
            (env, self._env_identifier), (model, self._training_param)
        ).save_trajectory()

        return model, plot

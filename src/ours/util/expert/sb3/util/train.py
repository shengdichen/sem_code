from gym import Env
from stable_baselines3 import PPO as PPOSB

from src.ours.eval.param import ExpertParam
from src.ours.util.common.helper import TqdmCallback
from src.ours.util.common.train import Trainer


class TrainerExpert(Trainer):
    def __init__(self, env: Env, training_param: ExpertParam):
        super().__init__(training_param)

        self._env = env

        self._model = PPOSB(
            "MlpPolicy",
            self._env,
            verbose=0,
            **self._training_param.kwargs_ppo,
            tensorboard_log=self._training_param.sb3_tblog_dir
        )

    @property
    def model(self):
        return self._model

    def train(self) -> None:
        self._model.learn(
            total_timesteps=self._training_param.n_steps_expert_train,
            callback=[TqdmCallback()],
        )

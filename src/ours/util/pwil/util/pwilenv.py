import numpy as np
from gym import Env

from src.ours.util.common.param import PwilParam
from src.upstream.env_utils import PWILReward


class PwilEnvFactory:
    def __init__(
        self,
        training_param: PwilParam,
        env_raw: Env,
        trajectories: list[np.ndarray],
    ):
        self._training_param = training_param
        self._env_raw = env_raw

        self._env_pwil_rewarded = self._make_env_pwil_rewarded(trajectories)

    @property
    def env_pwil_rewarded(self) -> Env:
        return self._env_pwil_rewarded

    def _make_env_pwil_rewarded(self, trajectories: list[np.ndarray]) -> Env:
        env_pwil_rewarded = PWILReward(
            env=self._env_raw,
            demos=trajectories,
            **self._training_param.pwil_training_param,
        )

        return env_pwil_rewarded

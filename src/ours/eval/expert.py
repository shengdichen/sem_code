from pathlib import Path

from gym import Env
from stable_baselines3 import PPO as PPOSB
from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.env.creation import PointEnvFactory, PathGenerator
from src.ours.eval.param import ExpertParam
from src.ours.util.helper import Plotter, TqdmCallback, ExpertManager
from src.ours.util.train import Trainer


class TrainerExpert(Trainer):
    def __init__(self, training_param: ExpertParam, env: Env):
        super().__init__(training_param)

        self._env = env
        self._demo_dir = training_param.demo_dir
        self._save_deterministic = False

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

    def train(self, n_timesteps, fname):
        self._model.learn(total_timesteps=n_timesteps, callback=[TqdmCallback()])

        ExpertManager.save_expert_traj(
            self._env,
            self._model,
            nr_trajectories=10,
            render=False,
            demo_dir=self._demo_dir,
            filename=fname + str(n_timesteps),
            deterministic=self._save_deterministic,
        )


class Saver:
    def __init__(self, model: BaseAlgorithm, savepath_rel: Path):
        self._model = model
        self._savepath_rel = savepath_rel

    def save_model(self):
        self._model.save(self._savepath_rel)


class ClientTrainerExpert:
    def __init__(self):
        self._training_param = ExpertParam()
        self._n_timesteps = self._training_param.n_steps_expert_train

    def train_and_plot(self) -> None:
        """
        # Train experts with different shifts representing their waypoint preferences
        """

        self._train_and_save({"n_targets": 2, "shift_x": 0, "shift_y": 0})
        self._train_and_save({"n_targets": 2, "shift_x": 0, "shift_y": 50})
        self._train_and_save({"n_targets": 2, "shift_x": 50, "shift_y": 0})

        self._plot()

    def _train_and_save(self, env_config: dict[str:int]) -> None:
        env = PointEnvFactory(env_config).create()
        trainer = TrainerExpert(self._training_param, env)
        filename = PathGenerator(env_config).get_filename_from_shift_values()

        trainer.train(self._n_timesteps, filename)
        Saver(
            trainer.model,
            Path(self._training_param.model_dir)
            / Path("model_" + filename + str(self._n_timesteps)),
        ).save_model()

    def _plot(self) -> None:
        Plotter.plot_experts(self._n_timesteps)
        Plotter.plot_experts(self._n_timesteps, hist=False)


def client_code():
    trainer = ClientTrainerExpert()
    trainer.train_and_plot()


if __name__ == "__main__":
    client_code()

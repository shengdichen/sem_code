import os

from gym import Env
from stable_baselines3 import PPO as PPOSB

from src.ours.env.creation import PointEnvFactory
from src.ours.eval.param import ExpertParam
from src.ours.util.helper import Plotter, TqdmCallback, ExpertManager
from src.ours.util.train import Trainer


class TrainerExpert(Trainer):
    def __init__(self, training_param: ExpertParam, env: Env):
        super().__init__(training_param)

        self._env = env
        self._model_dir = training_param.model_dir
        self._demo_dir = training_param.demo_dir
        self._save_deterministic = False

    def train(self, n_timesteps, fname):
        model = PPOSB(
            "MlpPolicy",
            self._env,
            verbose=0,
            **self._training_param.kwargs_ppo,
            tensorboard_log=self._training_param.sb3_tblog_dir
        )
        model.learn(total_timesteps=n_timesteps, callback=[TqdmCallback()])

        model.save(os.path.join(self._model_dir, "model_" + fname + str(n_timesteps)))
        ExpertManager.save_expert_traj(
            self._env,
            model,
            nr_trajectories=10,
            render=False,
            demo_dir=self._demo_dir,
            filename=fname + str(n_timesteps),
            deterministic=self._save_deterministic,
        )

        return model


class ClientTrainerExpert:
    def __init__(self):
        self._training_param = ExpertParam()
        self._n_timesteps = self._training_param.n_steps_expert_train

    def train_and_plot(self) -> None:
        """
        # Train experts with different shifts representing their waypoint preferences
        """

        self._train({"n_targets": 2, "shift_x": 0, "shift_y": 0})
        self._train({"n_targets": 2, "shift_x": 0, "shift_y": 50})
        self._train({"n_targets": 2, "shift_x": 50, "shift_y": 0})

        self._plot()

    def _train(self, env_config: dict[str:int]) -> None:
        env = PointEnvFactory(env_config).create()
        trainer = TrainerExpert(self._training_param, env)
        filename = self._get_filename_from_shift_values(env_config)

        trainer.train(self._n_timesteps, filename)

    @staticmethod
    def _get_filename_from_shift_values(env_conig: dict[str:int]) -> str:
        shift_x, shift_y = env_conig["shift_x"], env_conig["shift_y"]
        return "exp_" + str(shift_x) + "_" + str(shift_y)

    def _plot(self) -> None:
        Plotter.plot_experts(self._n_timesteps)
        Plotter.plot_experts(self._n_timesteps, hist=False)


def client_code():
    trainer = ClientTrainerExpert()
    trainer.train_and_plot()


if __name__ == "__main__":
    client_code()

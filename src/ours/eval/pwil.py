import os

import PIL.Image as Image
import matplotlib.pyplot as plt
import torchvision
from gym import Env
from stable_baselines3 import PPO, PPO as PPOSB
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from src.ours.env.creation import PointEnvFactory
from src.ours.env.env import MovePoint
from src.ours.eval.param import PwilParam
from src.ours.util.helper import ExpertManager, RewardPlotter, TqdmCallback
from src.ours.util.test import PolicyTester
from src.ours.util.train import Trainer
from src.upstream.env_utils import PWILReward
from src.upstream.utils import CustomCallback


class TrainerPwil(Trainer):
    def __init__(self, training_param: PwilParam, envs: tuple[Env, Env]):
        super().__init__(training_param)

        self._model_dir = "./models_pwil"
        self._save_deterministic = False

        self._env_raw, self._env_raw_testing = envs

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
            **self._kwargs_ppo,
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

        # save model
        if not os.path.exists(self._model_dir):
            os.mkdir(self._model_dir)

        model.save(os.path.join(self._model_dir, "model_" + fname + str(n_timesteps)))
        ExpertManager.save_expert_traj(
            env,
            model,
            nr_trajectories=10,
            render=False,
            filename=fname + str(n_timesteps),
            deterministic=self._save_deterministic,
        )

        return model, plot


class ClientTrainerPwil:
    def __init__(self):
        self._training_param = PwilParam()
        self._n_timesteps = int(3e5)

    def training(self):
        # train imitation learning / IRL policy
        demos = ExpertManager.load_expert_demos(self._n_timesteps)
        flat_demos = [item for sublist in demos for item in sublist]

        env_config = {"n_targets": 2, "shift_x": 0, "shift_y": 0}
        env_raw, env_raw_testing = (
            PointEnvFactory(env_config).create(),
            PointEnvFactory(env_config).create(),
        )
        trainer = TrainerPwil(self._training_param, (env_raw, env_raw_testing))
        model_pwil, plot = trainer.train(
            flat_demos,
            n_demos=3,
            subsampling=10,
            use_actions=False,
            n_timesteps=1e3,
            fname="pwil_0",
        )
        PolicyTester.test_policy("", model=model_pwil)
        im = Image.fromarray(plot)
        im.save("pwil.png")
        plt.figure()
        plt.imshow(im)

    def plot_grid(self):
        # plot grid of PWIL rewards
        plots = []
        demos = ExpertManager.load_expert_demos(self._n_timesteps)
        flat_demos_0 = [item for sublist in demos for item in sublist]
        flat_demos_01 = [item for sublist in demos[:1] for item in sublist]
        flat_demos_12 = [item for sublist in demos[1:] for item in sublist]

        for ss in [1, 2, 3, 5, 10, 20]:
            for j, dem in enumerate(
                [demos[0], flat_demos_01, flat_demos_12, flat_demos_0]
            ):
                for n_demos in [1, 2, 3]:
                    print("subsampling: ", ss, " dem: ", j, " n_demos: ", n_demos)
                    env = PWILReward(
                        env=MovePoint(2, 0, 0),
                        demos=dem,
                        n_demos=n_demos,
                        subsampling=ss,
                        use_actions=False,
                    )
                    plot = RewardPlotter.plot_reward(discriminator=None, env=env)
                    plots.append(plot)
                    im = Image.fromarray(plot)
                    im.save(
                        "pwil_plots/pwil_ss{}_demoidx{}_n_demos{}.png".format(
                            ss, j, n_demos
                        )
                    )

        torchvision.utils.save_image(plots, normalize=True, nrow=6)

    @staticmethod
    def test():
        save_dir = "models_pwil/model_pwil_0{}".format(1e3)
        model = PPO.load(save_dir)
        PolicyTester.test_policy("", model=model)


def client_code():
    trainer = ClientTrainerPwil()
    trainer.test()


if __name__ == "__main__":
    client_code()

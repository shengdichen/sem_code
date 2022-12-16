import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from stable_baselines3 import PPO

from src.ours.env.creation import (
    PointEnvFactory,
    PointEnvIdentifierGenerator,
    PointEnvConfigFactory,
)
from src.ours.env.env import MovePoint
from src.ours.eval.pointenv.expert import PointEnvExpertDefault
from src.ours.util.common.param import PwilParam
from src.ours.util.common.helper import RewardPlotter
from src.ours.util.pwil.train import (
    PwilManagerFactory,
    PwilManager,
)
from src.upstream.env_utils import PWILReward


class PointEnvPwilManagerFactory:
    def __init__(self):
        self._training_param = PwilParam()

        env_config = PointEnvConfigFactory().env_configs[0]
        self._env_raw, self._env_raw_testing = (
            PointEnvFactory(env_config).create(),
            PointEnvFactory(env_config).create(),
        )
        self._env_identifier = PointEnvIdentifierGenerator().from_env(self._env_raw)

        self._demos_all = self._get_all_demos()

    @staticmethod
    def _get_all_demos():
        pointenv_expert_default = PointEnvExpertDefault()

        demos = pointenv_expert_default._load()
        flat_demos = [item for sublist in demos for item in sublist]
        return flat_demos

    def get_manager_default(self) -> PwilManager:
        return PwilManagerFactory(
            self._training_param,
            ((self._env_raw, self._env_raw_testing), self._env_identifier),
            self._demos_all,
        ).pwil_manager


class ClientTrainerPwil:
    def __init__(self):
        self._training_param = PwilParam()

        env_config = PointEnvConfigFactory().env_configs[0]
        self._env_raw, self._env_raw_testing = (
            PointEnvFactory(env_config).create(),
            PointEnvFactory(env_config).create(),
        )
        self._env_identifier = PointEnvIdentifierGenerator().from_env(self._env_raw)

    def training(self):
        # train imitation learning / IRL policy
        pointenv_expert_default = PointEnvExpertDefault()
        demos = pointenv_expert_default._load()

        flat_demos = [item for sublist in demos for item in sublist]

        trainer = TrainerPwil(
            self._training_param,
            ((self._env_raw, self._env_raw_testing), self._env_identifier),
        )
        model_pwil, plot = trainer.train(
            flat_demos,
            n_demos=3,
            subsampling=10,
            use_actions=False,
            n_timesteps=self._training_param.n_steps_expert_train,
            fname="pwil_0",
        )
        PolicyTester.test_policy(model_pwil)
        im = Image.fromarray(plot)
        im.save("pwil.png")
        plt.figure()
        plt.imshow(im)

    def plot_grid(self):
        # plot grid of PWIL rewards
        plots = []
        pointenv_expert_default = PointEnvExpertDefault()
        demos = pointenv_expert_default._load()
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
                    np.save(
                        self._training_param.plot_dir
                        + "pwil_ss{}_demoidx{}_n_demos{}".format(ss, j, n_demos),
                        plot,
                    )

                    plots.append(plot)
                    im = Image.fromarray(plot)
                    im.save(
                        self._training_param.plot_dir
                        + "pwil_ss{}_demoidx{}_n_demos{}.png".format(ss, j, n_demos)
                    )

        torchvision.utils.save_image(plots, normalize=True, nrow=6)

    def test(self):
        save_dir = self._training_param.model_dir + "model_pwil_0{}".format(
            self._training_param.n_steps_expert_train
        )
        model = PPO.load(save_dir)
        PolicyTester.test_policy(model)


def client_code():
    trainer = ClientTrainerPwil()
    trainer.test()


if __name__ == "__main__":
    client_code()

import PIL.Image as Image
import matplotlib.pyplot as plt

from src.ours.env.env import MovePoint
from src.ours.eval.param import TrainingParam
from src.ours.util.helper import ExpertManager, RewardPlotter
from src.ours.util.test import PolicyTester
from src.ours.util.train import Training
from src.upstream.env_utils import PWILReward


class TrainerPwil:
    def __init__(self):
        self._training_param = TrainingParam()
        self._training = Training(self._training_param)

    def training(self):
        # train imitation learning / IRL policy
        train_pwil_ = True
        if train_pwil_:
            demos = ExpertManager.load_expert_demos(5e5)
            flat_demos = [item for sublist in demos for item in sublist]
            model_pwil, plot = self._training.train_pwil(
                flat_demos,
                n_demos=3,
                subsampling=10,
                use_actions=False,
                n_timesteps=1e3,
                n_targets=2,
                shift_x=0,
                shift_y=0,
                kwargs=self._training_param._kwargs,
                fname="pwil_0",
                model_dir="./models",
                save_deterministic=False,
            )
            PolicyTester.test_policy("", model=model_pwil)
            im = Image.fromarray(plot)
            im.save("pwil.png")
            plt.figure()
            plt.imshow(im)

        # plot grid of PWIL rewards
        plots = []
        demos = ExpertManager.load_expert_demos(5e5)
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
                    plots.append(RewardPlotter.plot_reward(discriminator=None, env=env))
                    im = Image.fromarray(plot)
                    im.save(
                        "pwil_plots/pwil_ss{}_demoidx{}_n_demos{}.png".format(
                            ss, j, n_demos
                        )
                    )

        # vutils.save_image(plots, normalize=True, nrow=6)

        # test_policy('', model=model_pwil)

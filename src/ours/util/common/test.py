import numpy as np
from matplotlib import pyplot as plt

from src.ours.env.env import MovePoint


class PolicyTester:
    @staticmethod
    def test_policy(
        model,
        rm="ERM",
        shift_x=0,
        shift_y=0,
        n_timesteps=2000,
        deterministic=True,
    ):
        testing_env = MovePoint(2, shift_x=shift_x, shift_y=shift_y)

        obs_list = []
        obs = testing_env.reset()
        rew = 0
        cum_rew = []

        for i in range(n_timesteps):
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, r, done, info = testing_env.step(action)
            rew += r
            if done:
                obs = testing_env.reset()
                print(rm + " rewards: ", rew)
                cum_rew.append(rew)

            obs_list.append(obs)

        print(rm + " mean / std cumrew: ", np.mean(cum_rew), np.std(cum_rew))
        obsa = np.stack(obs_list)

        fig = plt.figure()
        plt.title(rm)
        plt.plot(obsa[:, 0], obsa[:, 1], "m-", alpha=0.3)
        plt.scatter(obsa[:, 2], obsa[:, 3], c="r")
        plt.axis("equal")
        plt.show()

        return fig

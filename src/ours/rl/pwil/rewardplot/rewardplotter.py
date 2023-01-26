import numpy as np
import torch
from matplotlib import pyplot as plt


class RewardPlotter:
    @staticmethod
    def plot_reward(discriminator, plot_value=False, env=None):
        # generate grid w/ two different targets:
        x = np.arange(0, 200)
        y = np.arange(0, 200)

        grid = np.array(np.meshgrid(x, y))
        point_list = np.reshape(grid, [2, -1]).T

        tgt_pos_1 = np.array([100, 100])
        tgt_pos_2 = np.array([190, 176])

        grid_pos1 = np.concatenate(
            [point_list, np.tile(tgt_pos_1, (point_list.shape[0], 1))], 1
        )
        grid_pos2 = np.concatenate(
            [point_list, np.tile(tgt_pos_2, (point_list.shape[0], 1))], 1
        )

        def dist_f(grid_pos):
            return -(
                np.sqrt(
                    (grid_pos[:, 0] - grid_pos[:, 2]) ** 2
                    + (grid_pos[:, 1] - grid_pos[:, 3]) ** 2
                )
            )

        r_gt_1 = dist_f(grid_pos1).reshape((200, 200))
        r_gt_2 = dist_f(grid_pos2).reshape((200, 200))

        # plot rewards for both goals
        if env is None:
            with torch.no_grad():
                r_1 = discriminator.get_reward(ob=torch.Tensor(grid_pos1), ac=None)
                r_2 = discriminator.get_reward(ob=torch.Tensor(grid_pos2), ac=None)
                if plot_value:
                    v_1 = discriminator.get_value(ob=torch.Tensor(grid_pos1), ac=None)
                    v_2 = discriminator.get_value(ob=torch.Tensor(grid_pos2), ac=None)

                r_1 = r_1.numpy()
                r_2 = r_2.numpy()

        else:
            r1 = []
            r2 = []
            for pos in grid_pos1:
                r1.append(env.pwil.compute_reward(pos))
                env.pwil.reset()
            for pos in grid_pos2:
                r2.append(env.pwil.compute_reward(pos))
                env.pwil.reset()
            r_1 = np.stack(r1)
            r_2 = np.stack(r2)

        title = "reward"
        if plot_value:
            r_1 = v_1.numpy()
            r_2 = v_2.numpy()
            title = "value"

        fig = plt.gcf()
        fig.set_size_inches(28, 5)

        plt.subplot(151)
        for demo in env.pwil.demonstrations:
            plt.scatter(demo[:, 0], demo[:, 1], s=0.2, alpha=0.3)
        plt.title(title)
        plt.axis("equal")

        plt.subplot(152)
        plt.pcolor(x, y, r_1.reshape((200, 200)))
        plt.colorbar()
        plt.scatter(tgt_pos_1[0], tgt_pos_1[1], c="r")
        plt.title(title)
        plt.axis("equal")

        plt.subplot(153)
        plt.pcolor(x, y, r_gt_1)
        plt.colorbar()
        plt.scatter(tgt_pos_1[0], tgt_pos_1[1], c="r")
        plt.title(title)
        plt.axis("equal")

        plt.subplot(154)
        plt.pcolor(x, y, r_2.reshape((200, 200)))
        plt.colorbar()
        plt.scatter(tgt_pos_2[0], tgt_pos_2[1], c="r")
        plt.title(title)
        plt.axis("equal")

        plt.subplot(155)
        plt.pcolor(x, y, r_gt_2)
        plt.colorbar()
        plt.title(title)
        plt.scatter(tgt_pos_2[0], tgt_pos_2[1], c="r")
        plt.axis("equal")

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return image_from_plot

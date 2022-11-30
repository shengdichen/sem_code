import random

import numpy as np
from gym import Env, spaces

from src.ours.env.component.point import PointFactory
from src.ours.env.util import PointEnvRendererHuman, PointEnvRendererRgb


class MovePoint(Env):
    def __init__(self, n_targets=2, shift_x=0, shift_y=0, random_init=False):
        super(MovePoint, self).__init__()

        # Define a 2-D observation space
        self.canvas_shape = (200, 200, 3)
        self.observation_shape = 4
        self.observation_space = spaces.Box(
            low=np.zeros(self.observation_shape, dtype=np.float64),
            high=np.ones(self.observation_shape, dtype=np.float64) * 200,
            dtype=np.float64,
        )

        # Define an action space ranging from 0 to 4
        self.action_space = spaces.Discrete(
            5,
        )

        # Create a canvas to render the environment images upon
        self.canvas = np.ones(self.canvas_shape)
        self.canvas_hist = np.zeros(self.canvas_shape)

        # Define elements present inside the environment
        self.elements = []

        # Add targets
        self.n_tgt = n_targets
        self.targets = []

        # Maximum episode length
        self.max_time = 1000

        # Permissible area of helicper to be
        self.y_min = int(self.canvas_shape[0] * 0.1)
        self.x_min = 0
        self.y_max = int(self.canvas_shape[0] * 0.9)
        self.x_max = self.canvas_shape[1]
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.random_init = random_init

    @property
    def env_config(self):
        return {
            "n_targets": self.n_tgt,
            "shift_x": self.shift_x,
            "shift_y": self.shift_y,
        }

    def draw_elements_on_canvas(self):
        # Init the canvas
        self.canvas = np.ones(self.canvas_shape)

        # Draw the agent on canvas
        for elem in self.elements:
            x, y = elem.movement.x, elem.movement.y
            self.canvas[y : y + elem.y_icon, x : x + elem.x_icon] = elem.icon

        agent = self.agent
        self.canvas_hist[
            agent.movement.y : agent.movement.y + agent.y_icon,
            agent.movement.x : agent.movement.x + agent.x_icon,
        ] += 1

        # normalize hist canvas
        # self.canvas_hist = self.canvas_hist / np.sum(self.canvas_hist)
        # text = 'Time Left: {} | Rewards: {}'.format(self.time, self.ep_return)

        # Put the info on canvas
        # self.canvas = cv2.putText(
        #     self.canvas, text, (10, 20), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA
        # )

    def reset(self):
        # Flag that marks the termination of an episode
        self.done = False
        # Reset the fuel consumed
        self.time = self.max_time

        # Reset the reward
        self.curr_tgt = 0

        # Determine a place to intialise the agent in
        if self.random_init:
            x = random.randrange(
                int(self.canvas_shape[0] * 0.05), int(self.canvas_shape[0] * 0.10)
            )
            y = random.randrange(
                int(self.canvas_shape[1] * 0.15), int(self.canvas_shape[1] * 0.20)
            )
        else:
            x = 10
            y = 10

        self.elements = []
        # Intialise the agent
        self.agent = PointFactory(
            "agent", self.x_max, self.x_min, self.y_max, self.y_min
        ).create_agent()
        self.agent.movement.set_position(x, y)

        # Intialise the elements
        self.elements.append(self.agent)

        # Set the targets
        # self.targets = self.generate_targets()

        # define two targets to simulate different experts
        pos = [
            (
                int(self.canvas_shape[0] / 2) + self.shift_x,
                int(self.canvas_shape[1] / 2) + self.shift_y,
            ),
            (int(self.canvas_shape[0] * 0.95), int(self.canvas_shape[1] * 0.95)),
        ]
        self.targets = []
        for i, p in enumerate(pos):
            tgt = PointFactory(
                "tgt_{}".format(i), self.x_max, self.x_min, self.y_max, self.y_min
            ).create_target()
            tgt.movement.set_position(p[0], p[1])
            self.targets.append(tgt)

        self.elements.extend(self.targets)

        # Reset the Canvas
        self.canvas = np.ones(self.canvas_shape) * 1

        # Draw elements on the canvas
        self.draw_elements_on_canvas()

        curr_tgt = self.targets[self.curr_tgt]
        state = np.stack(
            [
                self.agent.movement.x,
                self.agent.movement.y,
                curr_tgt.movement.x,
                curr_tgt.movement.y,
            ]
        )

        # return the observation
        return state

    # TODO: expand to preferences as random process!
    def generate_random_targets(self):
        tgts = []
        for i in range(self.n_tgt):
            tgt = PointFactory(
                "tgt_{}".format(i), self.x_max, self.x_min, self.y_max, self.y_min
            ).create_target()

            tgt_x = random.randrange(
                self.y_min + int(self.y_max / 4), self.y_max - int(self.y_max / 4)
            )
            tgt_y = random.randrange(
                self.y_min + int(self.y_max / 4), self.y_max - int(self.y_max / 4)
            )
            tgt.movement.set_position(tgt_x, tgt_y)
            tgts.append(tgt)

        return tgts

    def step(self, action):
        # Assert that it is a valid action
        assert self.action_space.contains(action), "Invalid Action"

        # Decrease the time counter
        self.time -= 1

        # apply the action to the agent
        if action == 0:
            self.agent.movement.shift(0, 2)
        elif action == 1:
            self.agent.movement.shift(0, -2)
        elif action == 2:
            self.agent.movement.shift(2, 0)
        elif action == 3:
            self.agent.movement.shift(-2, 0)
        # REMOVE NOOP
        # elif action == 4:
        #    self.agent.move(0,0)

        curr_tgt = self.targets[self.curr_tgt]
        # l2 distance as reward
        reward = -(
            np.sqrt(
                (self.agent.movement.x - curr_tgt.movement.x) ** 2
                + (self.agent.movement.y - curr_tgt.movement.y) ** 2
            )
        )

        if self.agent.has_collided(curr_tgt):
            # reward += 5
            if self.curr_tgt == len(self.targets) - 1:
                # task solved
                # reward += 100
                self.done = True
            else:
                # update target
                self.curr_tgt += 1

        # Draw elements on the canvas
        self.draw_elements_on_canvas()

        state = np.stack(
            [
                self.agent.movement.x,
                self.agent.movement.y,
                curr_tgt.movement.x,
                curr_tgt.movement.y,
            ]
        )

        # If out of fuel, end the episode.
        if self.time == 0:
            self.done = True

        return state, reward, self.done, {}

    def render(self, mode="human") -> None:
        assert mode in [
            "human",
            "rgb_array",
        ], 'Invalid mode, must be either "human" or "rgb_array"'

        if mode == "human":
            renderer = PointEnvRendererHuman(self.canvas, self.canvas_hist)
        else:
            renderer = PointEnvRendererRgb(self.canvas)

        renderer.render()

    def close(self):
        PointEnvRendererHuman.clean_up()


def client_code():
    pointenv = MovePoint()
    pointenv.reset()
    pointenv.render("human")


if __name__ == "__main__":
    client_code()

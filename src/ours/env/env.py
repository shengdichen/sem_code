import random

import numpy as np
from gym import Env, spaces

from src.ours.env.canvas import TrajectoryHeatVisualizer, AgentTargetsVisualizer
from src.ours.env.component.point import PointFactory, NamedPointWithIcon
from src.ours.env.util import PointEnvRendererHuman, PointEnvRendererRgb


class MovePoint(Env):
    def __init__(self, n_targets=2, shift_x=0, shift_y=0, random_init=False):
        super(MovePoint, self).__init__()

        self._side_length = 200
        self.canvas_shape = self._side_length, self._side_length
        self.observation_shape = 4
        self.observation_space = spaces.Box(
            low=np.zeros(self.observation_shape, dtype=np.float64),
            high=np.ones(self.observation_shape, dtype=np.float64) * self._side_length,
            dtype=np.float64,
        )

        # Define an action space ranging from 0 to 4
        self.action_space = spaces.Discrete(
            5,
        )

        self._agent_targets_visualizer = AgentTargetsVisualizer(self.canvas_shape)
        self._trajectory_heat_visualizer = TrajectoryHeatVisualizer(self.canvas_shape)

        (
            self.y_min,
            self.x_min,
            self.y_max,
            self.x_max,
        ) = self._agent_targets_visualizer.get_movement_ranges()
        self.shift_x = shift_x
        self.shift_y = shift_y

        self.agent = self.make_agent()

        # Add targets
        self.n_tgt = n_targets
        self.curr_tgt_id = 0
        self.targets = self.make_targets()

        # Define elements present inside the environment
        self.agent_and_targets = []
        self.agent_and_targets.append(self.agent)
        self.agent_and_targets.extend(self.targets)

        # Maximum episode length
        self.max_time = 1000
        self.time = self.max_time

        self.random_init = random_init

        self.done = False

    @property
    def env_config(self):
        return {
            "n_targets": self.n_tgt,
            "shift_x": self.shift_x,
            "shift_y": self.shift_y,
        }

    def draw_elements_on_canvas(self):
        self._agent_targets_visualizer.register(self.agent_and_targets)
        self._trajectory_heat_visualizer.register(self.agent)

    def make_agent(self) -> NamedPointWithIcon:
        return PointFactory(
            "agent", self.x_max, self.x_min, self.y_max, self.y_min
        ).create_agent()

    def make_targets(self) -> list[NamedPointWithIcon]:
        targets = []
        for i in range(self.n_tgt):
            tgt = PointFactory(
                "tgt_{}".format(i), self.x_max, self.x_min, self.y_max, self.y_min
            ).create_target()
            targets.append(tgt)

        return targets

    def get_reset_agent_pos(self):
        if self.random_init:
            x, y = self._agent_targets_visualizer.get_reset_agent_pos_random()
        else:
            x = 10
            y = 10

        return x, y

    def get_reset_targets_pos(self):
        # define two targets to simulate different experts
        pos = [
            (
                int(self.canvas_shape[0] / 2) + self.shift_x,
                int(self.canvas_shape[1] / 2) + self.shift_y,
            ),
            (int(self.canvas_shape[0] * 0.95), int(self.canvas_shape[1] * 0.95)),
        ]
        return pos

    def reset(self):
        # Flag that marks the termination of an episode
        self.done = False
        # Reset the fuel consumed
        self.time = self.max_time

        # Determine a place to intialise the agent in
        x, y = self.get_reset_agent_pos()
        self.agent.movement.set_position(x, y)

        # Set the targets
        # self.targets = self.generate_targets()

        target_positions = self.get_reset_targets_pos()
        for target, target_pos in zip(self.targets, target_positions):
            target.movement.set_position(target_pos[0], target_pos[1])

        # Draw elements on the canvas
        self.draw_elements_on_canvas()

        # Reset the reward
        self.curr_tgt_id = 0

        obs = self._get_obs()
        return obs

    def _get_obs(self):
        state = np.stack(
            [
                self.agent.movement.x,
                self.agent.movement.y,
                self.targets[self.curr_tgt_id].movement.x,
                self.targets[self.curr_tgt_id].movement.y,
            ]
        )

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

    def step(self, action: int):
        # Decrease the time counter
        self.time -= 1

        shift = ActionConverter(action, self.action_space).get_shift()
        self.agent.movement.shift(shift[0], shift[1])

        reward = -1 * self.agent.distance_l2(self.targets[self.curr_tgt_id])

        if self.agent.has_collided(self.targets[self.curr_tgt_id]):
            # reward += 5
            if self.curr_tgt_id == len(self.targets) - 1:
                # task solved
                # reward += 100
                self.done = True
            else:
                # update target
                self.curr_tgt_id += 1

        # Draw elements on the canvas
        self.draw_elements_on_canvas()

        obs = self._get_obs()

        # If out of fuel, end the episode.
        if self.time == 0:
            self.done = True

        return obs, reward, self.done, {}

    def render(self, mode="human") -> None:
        assert mode in [
            "human",
            "rgb_array",
        ], 'Invalid mode, must be either "human" or "rgb_array"'

        if mode == "human":
            renderer = PointEnvRendererHuman(
                self._agent_targets_visualizer.colormat,
                self._trajectory_heat_visualizer.colormat,
            )
        else:
            renderer = PointEnvRendererRgb(self._agent_targets_visualizer.colormat)

        renderer.render()

    def close(self):
        PointEnvRendererHuman.clean_up()


class ActionConverter:
    def __init__(self, action: int, action_space: spaces.Space):
        assert action_space.contains(action), "Invalid Action"

        self._action = action

    def get_shift(self):
        if self._action == 0:
            shift = 0, 2
        elif self._action == 1:
            shift = 0, -2
        elif self._action == 2:
            shift = 2, 0
        elif self._action == 3:
            shift = -2, 0
        else:
            shift = 0, 0

        return shift

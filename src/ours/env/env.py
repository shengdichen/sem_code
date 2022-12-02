import numpy as np
from gym import Env, spaces

from src.ours.env.canvas import TrajectoryHeatVisualizer, AgentTargetsVisualizer
from src.ours.env.component.point import PointFactory, NamedPointWithIcon
from src.ours.env.space import SpacesGenerator
from src.ours.env.util import PointEnvRendererHuman, PointEnvRendererRgb


class MovePoint(Env):
    def __init__(self, n_targets=2, shift_x=0, shift_y=0, random_init=False):
        super(MovePoint, self).__init__()

        self._side_length = 200
        self.observation_space, self.action_space = SpacesGenerator(
            self._side_length
        ).get_spaces()

        self.canvas_shape = self._side_length, self._side_length
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

        self.random_init = random_init

        self.agent = self.make_agent()

        self.n_tgt = n_targets
        self.curr_tgt_id = 0
        self.targets = self.make_targets()

        self.agent_and_targets = []
        self.agent_and_targets.append(self.agent)
        self.agent_and_targets.extend(self.targets)

        self._max_episode_length, self._curr_episode_length = 1000, 0
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

    def make_targets(self, make_random_targets=False) -> list[NamedPointWithIcon]:
        targets = []
        for i in range(self.n_tgt):
            tgt = PointFactory(
                "tgt_{}".format(i), self.x_max, self.x_min, self.y_max, self.y_min
            ).create_target()

            # TODO: expand to preferences as random process!
            if make_random_targets:
                tgt_x, tgt_y = self._agent_targets_visualizer.get_target_pos_random()
                tgt.movement.set_position(tgt_x, tgt_y)

            targets.append(tgt)

        return targets

    def reset(self):
        x, y = self._agent_targets_visualizer.get_reset_agent_pos(self.random_init)
        self.agent.movement.set_position(x, y)

        target_positions = self._agent_targets_visualizer.get_reset_targets_pos(
            (self.shift_x, self.shift_y)
        )
        for target, target_pos in zip(self.targets, target_positions):
            target.movement.set_position(target_pos[0], target_pos[1])

        self.draw_elements_on_canvas()

        self.curr_tgt_id = 0

        self._curr_episode_length = 0
        self.done = False

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

    def step(self, action: int):
        shift = ActionConverter(action, self.action_space).get_shift()
        self.agent.movement.shift(shift[0], shift[1])

        reward = -1 * self.agent.distance_l2(self.targets[self.curr_tgt_id])

        self._update_target()

        self.draw_elements_on_canvas()

        obs = self._get_obs()

        self._curr_episode_length += 1
        if self._curr_episode_length == self._max_episode_length:
            self.done = True

        return obs, reward, self.done, {}

    def _update_target(self):
        if self.agent.has_collided(self.targets[self.curr_tgt_id]):
            # reward += 5
            if self.curr_tgt_id == len(self.targets) - 1:
                # task solved
                # reward += 100
                self.done = True
            else:
                self.curr_tgt_id += 1

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

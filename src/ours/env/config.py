class PointEnvConfigFactory:
    def __init__(self):
        self._env_configs = [
            {"n_targets": 2, "shift_x": 0, "shift_y": 0},
            {"n_targets": 2, "shift_x": 0, "shift_y": 50},
            {"n_targets": 2, "shift_x": 50, "shift_y": 0},
        ]

    @property
    def env_configs(self):
        return self._env_configs

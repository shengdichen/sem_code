from src.ours.env.env import PointNav


class PointEnvIdentifierGenerator:
    def __init__(self, prefix: str):
        self._prefix = prefix
        self._connector = "_"

    def get_identifier(self, env_config: dict[str:int]) -> str:
        shift_x, shift_y = env_config["shift_x"], env_config["shift_y"]
        return (
            self._prefix
            + self._connector
            + "{0:03}".format(shift_x)
            + self._connector
            + "{0:03}".format(shift_y)
        )

    def from_env(self, env: PointNav) -> str:
        return self.get_identifier(env.env_config)


class DiscretePointEnvIdentifierGenerator(PointEnvIdentifierGenerator):
    def __init__(self):
        super().__init__("pointenv")


class ContPointEnvIdentifierGenerator(PointEnvIdentifierGenerator):
    def __init__(self):
        super().__init__("pointenv_cont")

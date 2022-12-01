class TestEnv:
    @staticmethod
    def get_runner():
        from src.ours.env.run import PointEnvRunner

        return PointEnvRunner()

    def test_reset(self):
        self.get_runner().reset()

    def test_close(self):
        self.get_runner().close()

    def test_run_random(self):
        self.get_runner().run_random()

class TestEnv:
    @staticmethod
    def get_runner():
        from src.ours.eval.pointenv.run.run import PointEnvRunner

        return PointEnvRunner()

    def test_reset(self):
        self.get_runner().reset()

    def test_run_random(self):
        from src.ours.eval.pointenv.run.actionprovider import ActionProviderRandom

        self.get_runner().run_episodes(ActionProviderRandom())

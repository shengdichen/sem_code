class TestEnv:
    @staticmethod
    def get_runner():
        from src.ours.eval.pointenv.run.run import DiscretePointEnvRunner

        return DiscretePointEnvRunner()

    def test_reset(self):
        self.get_runner().reset()

    def test_run_random(self):
        from src.ours.eval.pointenv.run.actionprovider import ActionProviderRandom

        self.get_runner().run_episodes(ActionProviderRandom())


class TestContEnv:
    @staticmethod
    def get_runner():
        from src.ours.eval.pointenv.run.run import ContPointEnvRunner

        return ContPointEnvRunner()

    def test_reset(self):
        self.get_runner().reset()

    def test_run_random(self):
        from src.ours.eval.pointenv.run.actionprovider import ActionProviderRandomCont

        self.get_runner().run_episodes(ActionProviderRandomCont())

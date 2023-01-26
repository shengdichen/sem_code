class TestEnv:
    @staticmethod
    def get_runner():
        from src.ours.eval.pointenv.run.run import DiscretePointNavRunner

        return DiscretePointNavRunner()

    def test_reset(self):
        self.get_runner().reset()

    def test_run_random(self):
        from src.ours.eval.pointenv.run.action_provider import ActionProviderRandom

        self.get_runner().run_episodes(ActionProviderRandom())


class TestContEnv:
    @staticmethod
    def get_runner():
        from src.ours.eval.pointenv.run.run import ContPointNavRunner

        return ContPointNavRunner()

    def test_reset(self):
        self.get_runner().reset()

    def test_run_random(self):
        from src.ours.eval.pointenv.run.action_provider import ActionProviderRandomCont

        self.get_runner().run_episodes(ActionProviderRandomCont())

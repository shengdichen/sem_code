import gym
from stable_baselines3.common.base_class import BaseAlgorithm

from src.ours.util.common.param import CommonParam
from src.ours.util.expert.path import ExpertSaveLoadPathGenerator
from src.ours.util.common.saveload.path import SaveLoadPathGenerator
from src.ours.util.common.train import Trainer
from src.ours.util.expert.sb3.util.model import ExpertAlgorithmFactory
from src.ours.util.expert.sb3.util.saveload import Sb3Saver, Sb3Loader
from src.ours.util.expert.sb3.util.train import ExpertSb3Trainer


class Sb3Manager:
    def __init__(
        self,
        envs_and_identifier: tuple[tuple[gym.Env, gym.Env], str],
        path_generator: SaveLoadPathGenerator,
        algorithm: BaseAlgorithm,
    ):
        (self._env, self._env_eval), self._env_identifier = envs_and_identifier
        self._best_sb3_model_path = path_generator.get_best_sb3_model_path()
        self._latest_sb3_model_path = path_generator.get_latest_sb3_model_path()
        self._model = self._get_model(algorithm)

    def _get_model(self, algorithm: BaseAlgorithm) -> BaseAlgorithm:
        sb3_loader = Sb3Loader(algorithm, self._best_sb3_model_path)
        if sb3_loader.exists():
            return sb3_loader.load(self._env)
        else:
            return algorithm

    @property
    def model(self) -> BaseAlgorithm:
        return self._model

    def train(self) -> None:
        self._get_trainer().train()

    def _get_trainer(self) -> Trainer:
        pass

    def save(self) -> None:
        saver = Sb3Saver(self._model, self._latest_sb3_model_path)
        saver.save()


class ExpertSb3Manager(Sb3Manager):
    def __init__(
        self,
        envs_and_identifier: tuple[tuple[gym.Env, gym.Env], str],
        training_param: CommonParam,
    ):
        (env, __), env_identifier = envs_and_identifier

        super().__init__(
            envs_and_identifier,
            ExpertSaveLoadPathGenerator(env_identifier, training_param),
            ExpertAlgorithmFactory(
                (env, env_identifier), training_param
            ).get_algorithm(),
        )

        self._training_param = training_param

    def _get_trainer(self) -> ExpertSb3Trainer:
        return ExpertSb3Trainer(
            self._model, self._training_param, (self._env_eval, self._env_identifier)
        )

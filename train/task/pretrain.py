from omegaconf import DictConfig

from core.task import BaseTaskCls
from model.config import ModelConfig
from train.config import TrainConfig
from train.train import Trainer


class App(BaseTaskCls):

    def __init__(
        self,
        train_config: DictConfig,
        model_config: DictConfig,
        data_config: DictConfig,
        **kwargs,
    ):
        super(App, self).__init__(**kwargs)
        self.train_config = TrainConfig(**train_config)
        self.model_config = ModelConfig(**model_config)

    def run(self, **kwargs):
        #TODO(hyeontae): Implement run method
        pass

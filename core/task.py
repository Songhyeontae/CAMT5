import abc
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class BaseTaskCls(abc.ABC):

    def __init__(
        self,
        name: str,
        task_id: str,
        **kwargs,
    ):
        self.name = name
        self.task_id = task_id

    @abc.abstractmethod
    def run(self, **kwargs) -> Optional[List[str]]:
        pass


class DummyTask(BaseTaskCls):
    """Just for test and debugging"""

    def __init__(self, **kwargs):
        super(DummyTask, self).__init__(**kwargs)

    def run(self, **kwargs):
        logger.info("Invoke DummyApp.run")
        logger.info(f"kwargs: {kwargs}")

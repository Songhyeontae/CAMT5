import os

from dotenv import load_dotenv

load_dotenv()

if os.environ.get("HYDRA_FULL_ERROR") is None:
    os.environ["HYDRA_FULL_ERROR"] = "1"

import logging
import sys

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    app = instantiate(config.task)
    try:
        output = app.run()
        if output is not None and len(output) > 0:
            with open(config.output_path, "w") as f:
                f.write("\n".join(output))

    except Exception as e:
        logger.error("Failed to run the task.", exc_info=e)
        sys.exit(1)


if __name__ == "__main__":
    main()

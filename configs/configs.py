import logging
import os
from pytz import timezone
import yaml


def load_config() -> dict:
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_path, "configs/config.yaml")

    config_path_env = os.environ.get("CONFIG_FILE")
    if config_path_env is not None:
        config_path = config_path_env

    configs = yaml.safe_load(open(config_path))
    logging.info(f"Loaded config from {config_path}")

    # load_dotenv()
    # for key, value in os.environ.items():
    #     configs[key] = value

    return configs

vntz = timezone("Asia/Ho_Chi_Minh")

cfg = load_config()

"""Parsing configuration files for training scripts."""
import logging
import os

from omegaconf import OmegaConf

DEFAULT_CONFIG_FILENAME = "base.yaml"
MORE_CONFIGS_KEY = "mods"


def load_config():
    """Merge default config with more configs and overrides from CLI."""
    DEFAULT_CONFIG = OmegaConf.load(_build_path(DEFAULT_CONFIG_FILENAME))
    configs = [DEFAULT_CONFIG]

    CLI_CONFIG = OmegaConf.from_cli()
    if MORE_CONFIGS_KEY in CLI_CONFIG:
        for config_filepath in CLI_CONFIG[MORE_CONFIGS_KEY]:
            configs.append(OmegaConf.load(_build_path(config_filepath)))
            logging.info(f"Loaded config from {config_filepath}")
    configs.append(CLI_CONFIG)

    CONFIG = OmegaConf.merge(*configs)
    logging.info(OmegaConf.to_yaml(CONFIG))

    _validate_config(CONFIG)

    return CONFIG


def _validate_config(CONFIG):
    """Validate given config."""
    pass


def _build_path(filename):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)

"""
Python logging utils.
"""

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

import logging
from logging.handlers import TimedRotatingFileHandler

from colorlog import ColoredFormatter
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig


def get_logger(name: str = None, level: int = None) -> logging:
    """
    Get logger function.

    Args:
        name (str, optional): name of logging. Defaults to None.
        level (int, optional): logging level. Defaults to None.

    Returns:
        logging: logger function

    Examples:
        >>> logger = get_logger(__file__)
        >>> logger.debug("debug message")
        2023-01-03 12:00:00,000 [DEBUG] debug message | my_app.py:23
        >>> logger.info("info message")
        2023-01-03 12:00:00,000 [INFO] info message | my_app.py:23
    """

    # instantiate hydra config
    GlobalHydra.instance().clear()
    with initialize(config_path="../../config", version_base=None):
        cfg: DictConfig = compose(config_name="main")

    # set logging level
    level = cfg.logger.level if level is None else level

    # custom log level
    custom_level = {
        value["level"]: key for key, value in cfg.logger.custom_level.items()
    }
    custom_logging_level(custom_level)

    # custom log color
    custom_color = {
        key: value["color"] for key, value in cfg.logger.custom_level.items()
    }

    # rotating file handler to save logging file and
    # rotate every midnight and keep 30 days of logs
    rfh = TimedRotatingFileHandler(
        ROOT / "log/logs.log",
        when="midnight",
        backupCount=30,
    )
    rfh.setLevel(level)
    rfh.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s | %(filename)s:%(lineno)d"
        )
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(
        ColoredFormatter(
            "%(cyan)s%(asctime)s%(reset)s %(log_color)s[%(levelname)s] %(message)s | %(filename)s:%(lineno)d %(name)s",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "white",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold,red",
                **custom_color,
            },
        )
    )

    logging.basicConfig(
        level=level,
        handlers=[rfh, console_handler],
    )

    return logging


def custom_logging_level(custom_dict: dict) -> None:
    """
    Add custom logging level.

    Args:
        custom_dict (dict): dictionary of custom logging level.

    Examples:
        >>> custom_logging_level({21: "REDIS", 22: "MONGO", 23: "SQL", 24: "ENGINE"})
        >>> logger = get_logger(__file__)
        >>> logger.log(21, "redis message")
        2023-01-03 12:00:00,000 [REDIS] redis message | my_app.py:23
    """

    for key, value in custom_dict.items():
        logging.addLevelName(key, value)


if __name__ == "__main__":
    """Debugging."""
    import hydra

    logger = get_logger("Debugging")

    @hydra.main(config_path=f"{ROOT}/config", config_name="main", version_base=None)
    def main(cfg: DictConfig) -> None:
        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        logger.critical("critical message")
        logger.log(21, "redis message")
        logger.log(22, "mongo message")
        logger.log(23, "sql message")
        logger.log(24, "engine message")

    main()
import torch
import logging

logger = logging.getLogger()


def rank0_log(msg):
    if torch.distributed.get_rank() == 0:
        logger.info(msg)


def init_logger():
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

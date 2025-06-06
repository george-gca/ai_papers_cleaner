import logging
from logging.handlers import RotatingFileHandler
from multiprocessing import cpu_count
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map


SUPPORTED_CONFERENCES = [
    'aaai',
    'acl',
    'coling',
    'cvpr',
    'eacl',
    'eccv',
    'emnlp',
    'findings',
    'iccv',
    'iclr',
    'icml',
    'icra',
    'ijcai',
    'ijcnlp',
    'ijcv',
    'kdd',
    'naacl',
    'neurips',
    'neurips_workshop',
    'sigchi',
    'sigdial',
    'siggraph',
    'siggraph-asia',
    'tacl',
    'tpami',
    'wacv',
]


def parallelize_dataframe(df: pd.DataFrame, func: Callable, n_processes: int = cpu_count() // 4) -> pd.DataFrame:
    df_split = np.array_split(df, n_processes)
    results = process_map(func, df_split, max_workers=n_processes)
    df = pd.concat(results)
    return df


def setup_log(
        log_level: str = 'warning',
        log_file: str | Path = Path('run.log'),
        file_log_level: str = 'info',
        logs_to_silence: list[str] = [],
        ) -> None:
    """
    Setup the logging.

    Args:
        log_level (str): stdout log level. Defaults to 'warning'.
        log_file (str | Path): file where the log output should be stored. Defaults to 'run.log'.
        file_log_level (str): file log level. Defaults to 'info'.
        logs_to_silence (list[str]): list of loggers to be silenced. Useful when using log level < 'warning'. Defaults to [].
    """
    # TODO: fix this according to this
    # https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
    # https://www.electricmonk.nl/log/2017/08/06/understanding-pythons-logging-module/
    logging.PRINT = 60
    logging.addLevelName(60, 'PRINT')

    def log_print(self, message, *args, **kws):
        if self.isEnabledFor(logging.PRINT):
            # Yes, logger takes its '*args' as 'args'.
            self._log(logging.PRINT, message, args, **kws)

    logging.Logger.print = log_print

    # convert log levels to int
    int_log_level = {
        'debug': logging.DEBUG,  # 10
        'info': logging.INFO,  # 20
        'warning': logging.WARNING,  # 30
        'error': logging.ERROR,  # 40
        'critical': logging.CRITICAL,  # 50
        'print': logging.PRINT,  # 60
    }

    log_level = int_log_level[log_level]
    file_log_level = int_log_level[file_log_level]

    # create a handler to log to stderr
    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(log_level)

    # create a logging format
    if log_level >= logging.WARNING:
        stderr_formatter = logging.Formatter('{message}', style='{')
    else:
        stderr_formatter = logging.Formatter(
            # format:
            # <10 = pad with spaces if needed until it reaches 10 chars length
            # .10 = limit the length to 10 chars
            '{name:<10.10} [{levelname:.1}] {message}', style='{')
    stderr_handler.setFormatter(stderr_formatter)

    # create a file handler that have size limit
    if isinstance(log_file, str):
        log_file = Path(log_file).expanduser()

    file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5)  # ~ 5 MB
    file_handler.setLevel(file_log_level)

    # https://docs.python.org/3/library/logging.html#logrecord-attributes
    file_formatter = logging.Formatter(
        '{asctime} - {name:<12.12} {levelname:<8} {message}', datefmt='%Y-%m-%d %H:%M:%S', style='{')
    file_handler.setFormatter(file_formatter)

    # add the handlers to the root logger
    logging.basicConfig(handlers=[file_handler, stderr_handler], level=logging.DEBUG)

    # change logger level of logs_to_silence to warning
    for other_logger in logs_to_silence:
        logging.getLogger(other_logger).setLevel(logging.WARNING)

    # create logger
    logger = logging.getLogger(__name__)

    logger.info(f'Saving logs to {log_file.absolute()}')
    logger.info(f'Log level: {logging.getLevelName(log_level)}')

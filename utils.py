import logging
from logging.handlers import RotatingFileHandler
from multiprocessing import cpu_count
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map


supported_conferences = [
    'aaai/2017',
    'aaai/2018',
    'aaai/2019',
    'aaai/2020',
    'aaai/2021',
    'aaai/2022',
    'acl/2017',
    'acl/2018',
    'acl/2019',
    'acl/2020',
    'acl/2021',
    'acl/2022',
    'acl/2023',
    'coling/2018',
    'coling/2020',
    'coling/2022',
    'cvpr/2017',
    'cvpr/2018',
    'cvpr/2019',
    'cvpr/2020',
    'cvpr/2021',
    'cvpr/2022',
    'cvpr/2023',
    'eacl/2017',
    'eacl/2021',
    'eacl/2023',
    'eccv/2018',
    'eccv/2020',
    'eccv/2022',
    'emnlp/2017',
    'emnlp/2018',
    'emnlp/2019',
    'emnlp/2020',
    'emnlp/2021',
    'emnlp/2022',
    'findings/2020',
    'findings/2021',
    'findings/2022',
    'findings/2023',
    'iccv/2017',
    'iccv/2019',
    'iccv/2021',
    'iclr/2018',
    'iclr/2019',
    'iclr/2020',
    'iclr/2021',
    'iclr/2022',
    'iclr/2023',
    'icml/2017',
    'icml/2018',
    'icml/2019',
    'icml/2020',
    'icml/2021',
    'icml/2022',
    'icml/2023',
    'ijcai/2017',
    'ijcai/2018',
    'ijcai/2019',
    'ijcai/2020',
    'ijcai/2021',
    'ijcai/2022',
    'ijcnlp/2017',
    'ijcnlp/2019',
    'ijcnlp/2021',
    'ijcnlp/2022',
    'kdd/2017',
    'kdd/2018',
    'kdd/2020',
    'kdd/2021',
    'kdd/2022',
    'naacl/2018',
    'naacl/2019',
    'naacl/2021',
    'naacl/2022',
    'neurips/2017',
    'neurips/2018',
    'neurips/2019',
    'neurips/2020',
    'neurips/2021',
    'neurips/2022',
    'neurips_workshop/2019',
    'neurips_workshop/2020',
    'neurips_workshop/2021',
    'neurips_workshop/2022',
    'sigchi/2018',
    'sigchi/2019',
    'sigchi/2020',
    'sigchi/2021',
    'sigchi/2022',
    'sigdial/2017',
    'sigdial/2018',
    'sigdial/2019',
    'sigdial/2020',
    'sigdial/2021',
    'sigdial/2022',
    'tacl/2017',
    'tacl/2018',
    'tacl/2019',
    'tacl/2020',
    'tacl/2021',
    'tacl/2022',
    'tacl/2023',
    'wacv/2020',
    'wacv/2021',
    'wacv/2022',
    'wacv/2023',
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

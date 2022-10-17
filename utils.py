import argparse
import logging
from multiprocessing import cpu_count
from pathlib import Path
from typing import Callable, Union

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
    'coling/2018',
    'coling/2020',
    'coling/2022',
    'cvpr/2017',
    'cvpr/2018',
    'cvpr/2019',
    'cvpr/2020',
    'cvpr/2021',
    'cvpr/2022',
    'eacl/2017',
    'eacl/2021',
    'eccv/2018',
    'eccv/2020',
    'emnlp/2017',
    'emnlp/2018',
    'emnlp/2019',
    'emnlp/2020',
    'emnlp/2021',
    'findings/2020',
    'findings/2021',
    'findings/2022',
    'iccv/2017',
    'iccv/2019',
    'iccv/2021',
    'iclr/2018',
    'iclr/2019',
    'iclr/2020',
    'iclr/2021',
    'iclr/2022',
    'icml/2017',
    'icml/2018',
    'icml/2019',
    'icml/2020',
    'icml/2021',
    'icml/2022',
    'ijcai/2017',
    'ijcai/2018',
    'ijcai/2019',
    'ijcai/2020',
    'ijcai/2021',
    'ijcai/2022',
    'ijcnlp/2017',
    'ijcnlp/2019',
    'ijcnlp/2021',
    'kdd/2017',
    'kdd/2018',
    'kdd/2020',
    'kdd/2021',
    'naacl/2018',
    'naacl/2019',
    'naacl/2021',
    'naacl/2022',
    'neurips/2017',
    'neurips/2018',
    'neurips/2019',
    'neurips/2020',
    'neurips/2021',
    'neurips_workshop/2019',
    'neurips_workshop/2020',
    'neurips_workshop/2021',
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
    'tacl/2017',
    'tacl/2018',
    'tacl/2019',
    'tacl/2020',
    'tacl/2021',
    'tacl/2022',
    'wacv/2020',
    'wacv/2021',
    'wacv/2022',
]


conferences_pdfs = [c for c in supported_conferences if not c.startswith('kdd') and not c.startswith('sigchi')]


def parallelize_dataframe(df: pd.DataFrame, func: Callable, n_processes: int = cpu_count() // 4) -> pd.DataFrame:
    df_split = np.array_split(df, n_processes)
    results = process_map(func, df_split, max_workers=n_processes)
    df = pd.concat(results)
    return df


def recreate_url(url_str: str, conference: str, year: int, is_abstract: bool = False) -> str:
    if url_str == None or len(url_str) == 0:
        return ''

    if url_str.startswith('http://') or url_str.startswith('https://'):
        return url_str

    conference_lower = conference.lower()
    supported_conferences_names = set(
        [c.split('/')[0] for c in supported_conferences] + ['arxiv'])
    assert conference_lower in supported_conferences_names, f'conference is {conference}'

    if conference_lower == 'aaai':
        if year <= 2018:
            return f'https://www.aaai.org/ocs/index.php/AAAI/AAAI{year % 2000}/paper/viewPaper/{url_str}'
        else:
            return f'https://ojs.aaai.org/index.php/AAAI/article/view/{url_str}'

    # acl conferences
    elif conference_lower in {'acl', 'coling', 'eacl', 'emnlp', 'findings', 'ijcnlp', 'naacl', 'sigdial', 'tacl'}:
        return f'https://aclanthology.org/{url_str}'

    # arxiv
    elif conference_lower == 'arxiv':
        if is_abstract:
            url_type = 'abs'
            url_ext = ''
        else:
            url_type = 'pdf'
            url_ext = '.pdf'

        return f'https://arxiv.org/{url_type}/{url_str}{url_ext}'

    # thecvf conferences
    elif conference_lower in {'cvpr', 'iccv', 'wacv'}:
        return f'https://openaccess.thecvf.com/{url_str}'

    elif conference_lower == 'eccv':
        if is_abstract:
            url_type = 'html'
            url_ext = '.php'
        else:
            url_type = 'papers'
            url_ext = '.pdf'

        return f'https://www.ecva.net/papers/eccv_{year}/papers_ECCV/{url_type}/{url_str}{url_ext}'

    elif conference_lower in {'iclr', 'neurips_workshop'}:
        if is_abstract:
            url_type = 'forum'
        else:
            url_type = 'pdf'

        return f'https://openreview.net/{url_type}?id={url_str}'

    elif conference_lower == 'icml':
        if is_abstract:
            url_ext = '.html'
        else:
            url_ext = f'/{url_str.split("/")[1]}.pdf'

        return f'http://proceedings.mlr.press/{url_str}{url_ext}'

    elif conference_lower == 'ijcai':
        return f'https://www.ijcai.org/proceedings/{year}/{url_str}'

    elif conference_lower == 'kdd':
        if year == 2017:
            return f'https://www.kdd.org/kdd{year}/papers/view/{url_str}'
        elif year == 2018 or year == 2020:
            return f'https://www.kdd.org/kdd{year}/accepted-papers/view/{url_str}'
        else: # if year == 2021:
            return f'https://dl.acm.org/doi/abs/{url_str}'

    elif conference_lower == 'neurips':
        if is_abstract:
            url_type = 'hash'
        else:
            url_type = 'file'

        return f'https://papers.nips.cc/paper/{year}/{url_type}/{url_str}'

    elif conference_lower == 'sigchi':
        return f'https://dl.acm.org/doi/abs/{url_str}'

    return url_str


def setup_log(args: argparse.Namespace, log_file: Union[str, Path] = Path('run.log'), file_log_level: int = logging.INFO, logs_to_silence: list[str] = []) -> logging.Logger:
    """
    Setup the logging.

    Args:
        args (argparse.Namespace): args passed when calling the code (as in argparse)
        name (str): name of the created logger
        log_file (Union[str, Path], optional): file where the log output should be stored. Defaults to 'run.log'.
        logs_to_silence (list[str], optional): list of loggers to be silenced. Useful when using log level < logging.WARNING. Defaults to [].

    Returns:
        logging.Logger: default logger with given name
    """
    logging.PRINT = 60
    logging.addLevelName(60, 'PRINT')

    def log_print(self, message, *args, **kws):
        if self.isEnabledFor(logging.PRINT):
            # Yes, logger takes its '*args' as 'args'.
            self._log(logging.PRINT, message, args, **kws)

    logging.Logger.print = log_print

    log_level = {
        'debug': logging.DEBUG,  # 10
        'info': logging.INFO,  # 20
        'warning': logging.WARNING,  # 30
        'error': logging.ERROR,  # 40
        'critical': logging.CRITICAL,  # 50
        'print': logging.PRINT,  # 60
    }[args.log_level]

    # create a handler to log to stderr
    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(log_level)

    # create a logging format
    # if log_level >= logging.WARNING:
    stderr_formatter = logging.Formatter('{message}', style='{')
    # else:
        # stderr_formatter = logging.Formatter(
        #     # format:
        #     # <10 = pad with spaces if needed until it reaches 10 chars length
        #     # .10 = limit the length to 10 chars
        #     '{name:<10.10} [{levelname:.1}] {message}', style='{')
    stderr_handler.setFormatter(stderr_formatter)

    # create a file handler that have size limit
    # file_handler = logging.handlers.RotatingFileHandler(os.path.join(
    #     logdir, 'log.txt'), maxBytes=5000000, backupCount=5)  # ~ 5 MB

    # create a handler to log to file
    if isinstance(log_file, str):
        log_file = Path(log_file).expanduser()

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(file_log_level)

    # https://docs.python.org/3/library/logging.html#logrecord-attributes
    file_formatter = logging.Formatter(
        '{asctime} - {name:<12.12} {levelname:<8} {message}', datefmt='%Y-%m-%d %H:%M:%S', style='{')
    file_handler.setFormatter(file_formatter)

    # add the handlers to the root logger
    logging.basicConfig(handlers=[file_handler, stderr_handler], level=log_level)

    # change logger level of logs_to_silence to warning
    for other_logger in logs_to_silence:
        logging.getLogger(other_logger).setLevel(logging.WARNING)

    # create logger
    logger = logging.getLogger(__name__)

    logger.info(f'Saving logs to {log_file.absolute()}')
    logger.info(f'Log level: {logging.getLevelName(log_level)}')

    return logger

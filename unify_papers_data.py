import argparse
import locale
import logging
from pathlib import Path

import pandas as pd

from utils import setup_log, supported_conferences


_logger = logging.getLogger(__name__)
locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')


def _concat_filtered_df(joined_df: pd.DataFrame, file: Path, conference: str, year: str, sep: str, titles_already_in: set[str]) -> pd.DataFrame:
    df = pd.read_csv(file, sep=sep, dtype=str, keep_default_na=False)

    if len(titles_already_in) > 0:
        df = df[~df['title'].isin(titles_already_in)]

    df['conference'] = conference
    df['year'] = year
    return pd.concat([joined_df, df], ignore_index=True)


def main(args):
    log_dir = Path('logs/').expanduser()
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / 'unify_papers_data.log'
    setup_log(args, log_file)

    data_dir = Path(args.data_dir).expanduser()
    abstract_sep = '|'
    paper_info_sep = ';'

    # join papers' informations from all conferences
    conferences = [c for c in supported_conferences if (data_dir / c).exists()]
    conference = conferences[0]
    conf, year = conference.split('/')

    abstracts_file = data_dir / conference / 'abstracts.csv'
    joined_abstracts = pd.read_csv(abstracts_file, sep=abstract_sep, dtype=str, keep_default_na=False)
    joined_abstracts['conference'] = conf
    joined_abstracts['year'] = year

    abstracts_clean_file = data_dir / conference / 'abstracts_clean.csv'
    joined_abstracts_clean = pd.read_csv(abstracts_clean_file, sep=abstract_sep, dtype=str, keep_default_na=False)
    joined_abstracts_clean['conference'] = conf
    joined_abstracts_clean['year'] = year

    papers_info_file = data_dir / conference / 'paper_info.csv'
    joined_paper_info = pd.read_csv(papers_info_file, sep=paper_info_sep, dtype=str, keep_default_na=False)
    joined_paper_info['conference'] = conf
    joined_paper_info['year'] = year

    pdfs_urls_file = data_dir / conference / 'pdfs_urls.csv'
    joined_pdfs_urls = pd.read_csv(pdfs_urls_file, sep=paper_info_sep, dtype=str, keep_default_na=False)
    joined_pdfs_urls['conference'] = conf
    joined_pdfs_urls['year'] = year

    joined_papers_titles = set(joined_abstracts['title'])

    for conference in conferences[1:]:
        conf, year = conference.split('/')

        # adding new abstracts from this conference that have not been added yet
        abstracts_file = data_dir / conference / 'abstracts.csv'

        if not abstracts_file.exists():
            continue

        df = pd.read_csv(abstracts_file, sep=abstract_sep, dtype=str, keep_default_na=False)

        papers_titles = set(df['title'])
        papers_already_joined = papers_titles.intersection(joined_papers_titles)

        if len(papers_already_joined) > 0:
            _logger.warning(f'{len(papers_already_joined)} papers already joined from {conf} {year} out of {len(papers_titles)}')
            df = df[~df['title'].isin(papers_already_joined)]

        df['conference'] = conf
        df['year'] = year
        joined_abstracts = pd.concat([joined_abstracts, df], ignore_index=True)

        # adding new abstracts_clean from this conference that have not been added yet
        joined_abstracts_clean = _concat_filtered_df(joined_abstracts_clean, data_dir / conference / 'abstracts_clean.csv', conf, year, abstract_sep, papers_already_joined)

        # adding new paper_info from this conference that have not been added yet
        joined_paper_info = _concat_filtered_df(joined_paper_info, data_dir / conference / 'paper_info.csv', conf, year, paper_info_sep, papers_already_joined)

        # adding new pdfs_urls from this conference that have not been added yet
        joined_pdfs_urls = _concat_filtered_df(joined_pdfs_urls, data_dir / conference / 'pdfs_urls.csv', conf, year, abstract_sep, papers_already_joined)

        joined_papers_titles.update(papers_titles)

    joined_abstracts.to_feather(data_dir / 'abstracts.feather', compression='zstd')
    joined_abstracts_clean.to_feather(data_dir / 'abstracts_clean.feather', compression='zstd')
    joined_paper_info.to_feather(data_dir / 'paper_info.feather', compression='zstd')
    joined_pdfs_urls.to_feather(data_dir / 'pdfs_urls.feather', compression='zstd')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data',
                        help='directory for the input data')
    parser.add_argument('-l', '--log_level', type=str, default='warning',
                        choices=('debug', 'info', 'warning',
                                 'error', 'critical', 'print'),
                        help='log level to debug')

    args = parser.parse_args()

    main(args)

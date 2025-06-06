import argparse
import datetime
import locale
import logging
from pathlib import Path

import ftfy
import pandas as pd
from tqdm import tqdm

from utils import setup_log, SUPPORTED_CONFERENCES


_logger = logging.getLogger(__name__)
locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')


def _concat_filtered_df(joined_df: pd.DataFrame, file: Path, conference: str, year: str, sep: str, titles_already_in: set[str]) -> pd.DataFrame:
    df = pd.read_csv(file, sep=sep, dtype=str, keep_default_na=False)
    df['title'] = df['title'].apply(ftfy.fix_text)

    if len(titles_already_in) > 0:
        df = df[~df['title'].isin(titles_already_in)]

    df['conference'] = conference
    df['year'] = year
    return pd.concat([joined_df, df], ignore_index=True)


def main(args):
    log_dir = Path('logs/').expanduser()
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / 'unify_papers_data.log'
    setup_log(args.log_level, log_file)

    data_dir = Path(args.data_dir).expanduser()
    abstract_sep = '\t'
    paper_info_sep = '\t'

    # join papers' informations from all conferences
    conferences = [f'{c}/{y}' for c in SUPPORTED_CONFERENCES for y in range(2017, datetime.date.today().year + 1) if (data_dir / f'{c}/{y}').exists()]
    conference = conferences[0]
    conf, year = conference.split('/')

    abstracts_file = data_dir / conference / 'abstracts.tsv'
    joined_abstracts = pd.read_csv(abstracts_file, sep=abstract_sep, dtype=str, keep_default_na=False)
    joined_abstracts['conference'] = conf
    joined_abstracts['year'] = year
    joined_abstracts['title'] = joined_abstracts['title'].apply(ftfy.fix_text)

    abstracts_clean_file = data_dir / conference / 'abstracts_clean.tsv'
    joined_abstracts_clean = pd.read_csv(abstracts_clean_file, sep=abstract_sep, dtype=str, keep_default_na=False)
    joined_abstracts_clean['conference'] = conf
    joined_abstracts_clean['year'] = year
    joined_abstracts_clean['title'] = joined_abstracts_clean['title'].apply(ftfy.fix_text)

    papers_info_file = data_dir / conference / 'paper_info.tsv'
    joined_paper_info = pd.read_csv(papers_info_file, sep=paper_info_sep, dtype=str, keep_default_na=False)
    joined_paper_info['conference'] = conf
    joined_paper_info['year'] = year
    joined_paper_info['title'] = joined_paper_info['title'].apply(ftfy.fix_text)

    pdfs_urls_file = data_dir / conference / 'pdfs_urls.tsv'
    joined_pdfs_urls = pd.read_csv(pdfs_urls_file, sep=abstract_sep, dtype=str, keep_default_na=False)
    joined_pdfs_urls['conference'] = conf
    joined_pdfs_urls['year'] = year
    joined_pdfs_urls['title'] = joined_pdfs_urls['title'].apply(ftfy.fix_text)

    if not (len(joined_abstracts) == len(joined_abstracts_clean) == len(joined_paper_info)):
        _logger.error(f'Number of papers information after {conf} {year} differ: {len(joined_abstracts)}, {len(joined_abstracts_clean)}, and {len(joined_paper_info)}')
        raise ValueError(f'Number of papers information after {conf} {year} differ: {len(joined_abstracts)}, {len(joined_abstracts_clean)}, and {len(joined_paper_info)}')

    joined_papers_titles = set(joined_abstracts['title'])

    with tqdm(conferences[1:]) as pbar:
        for conference in pbar:
            conf, year = conference.split('/')
            pbar.set_description(f'{conf} {year}')

            # adding new abstracts from this conference that have not been added yet
            abstracts_file = data_dir / conference / 'abstracts.tsv'

            if not abstracts_file.exists():
                _logger.error(f'File not found: {abstracts_file}')
                continue

            try:
                df = pd.read_csv(abstracts_file, sep=abstract_sep, dtype=str, keep_default_na=False)
            except Exception as e:
                _logger.error(f'Failed to read {abstracts_file}')
                continue
                # raise e

            df['title'] = df['title'].apply(ftfy.fix_text)
            papers_titles = set(df['title'])
            papers_already_joined = papers_titles.intersection(joined_papers_titles)

            if len(papers_already_joined) > 0:
                if len(papers_already_joined) == len(papers_titles):
                    _logger.warning(f'All {len(papers_already_joined)} papers from {conf} {year} already joined')
                    continue
                else:
                    _logger.warning(f'{len(papers_already_joined)} papers already joined from {conf} {year} out of {len(papers_titles)}')

                    if len(papers_already_joined) <= 10:
                        for paper in papers_already_joined:
                            _logger.info(f'\t{paper}')

                if _logger.isEnabledFor(logging.DEBUG):
                    for title in papers_already_joined:
                        _logger.debug(f'\t{title}')

                df = df[~df['title'].isin(papers_already_joined)]
                tqdm.write(f'Remaining len(df): {len(df)}')

            df['conference'] = conf
            df['year'] = year
            joined_abstracts = pd.concat([joined_abstracts, df], ignore_index=True)

            # adding new abstracts_clean from this conference that have not been added yet
            joined_abstracts_clean = _concat_filtered_df(joined_abstracts_clean, data_dir / conference / 'abstracts_clean.tsv', conf, year, abstract_sep, papers_already_joined)

            # adding new paper_info from this conference that have not been added yet
            joined_paper_info = _concat_filtered_df(joined_paper_info, data_dir / conference / 'paper_info.tsv', conf, year, paper_info_sep, papers_already_joined)

            # adding new pdfs_urls from this conference that have not been added yet
            if (data_dir / conference / 'pdfs_urls.tsv').exists():
                joined_pdfs_urls = _concat_filtered_df(joined_pdfs_urls, data_dir / conference / 'pdfs_urls.tsv', conf, year, abstract_sep, papers_already_joined)

            joined_papers_titles.update(papers_titles)

            if not (len(joined_abstracts) == len(joined_abstracts_clean) == len(joined_paper_info)):
                _logger.error(f'Number of papers information after {conf} {year} differ: {len(joined_abstracts)}, {len(joined_abstracts_clean)}, and {len(joined_paper_info)}')

                # find the dataframe that has the minimum length and the one that has the maximum length, then log the titles of the papers that are missing
                min_length = min(len(joined_abstracts), len(joined_abstracts_clean), len(joined_paper_info))
                max_length = max(len(joined_abstracts), len(joined_abstracts_clean), len(joined_paper_info))

                if len(joined_abstracts) == min_length:
                    min_df = joined_abstracts
                    min_df_name = 'abstracts'
                elif len(joined_abstracts_clean) == min_length:
                    min_df = joined_abstracts_clean
                    min_df_name = 'abstracts_clean'
                else:
                    min_df = joined_paper_info
                    min_df_name = 'paper_info'

                if len(joined_abstracts) == max_length:
                    max_df = joined_abstracts
                elif len(joined_abstracts_clean) == max_length:
                    max_df = joined_abstracts_clean
                else:
                    max_df = joined_paper_info

                missing_titles = "\n".join(set(max_df['title']) - set(min_df['title']))
                _logger.error(f'Missing titles in {min_df_name}:\n{missing_titles}')
                raise ValueError(f'Number of papers information after {conf} {year} differ: {len(joined_abstracts)}, {len(joined_abstracts_clean)}, and {len(joined_paper_info)}')

    _logger.info(f'Final sizes:\n\tabstracts: {len(joined_abstracts):n}\n\tabstracts_clean: {len(joined_abstracts_clean):n}\n\tpaper_info: {len(joined_paper_info):n}\n\tpdfs_urls: {len(joined_pdfs_urls):n}')

    joined_abstracts.to_feather(data_dir / 'abstracts.feather', compression='zstd')
    joined_abstracts_clean.to_feather(data_dir / 'abstracts_clean.feather', compression='zstd')
    joined_paper_info.to_feather(data_dir / 'paper_info.feather', compression='zstd')
    joined_pdfs_urls.to_feather(data_dir / 'pdfs_urls.feather', compression='zstd')

    if not (len(joined_abstracts_clean) == len(joined_paper_info)):
        _logger.error(f'{len(joined_abstracts_clean)} abstracts clean and {len(joined_paper_info)} papers infos')
        raise ValueError(f'{len(joined_abstracts_clean)} abstracts clean and {len(joined_paper_info)} papers infos')


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

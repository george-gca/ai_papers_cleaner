import argparse
import logging
import multiprocessing
import re
from ast import literal_eval
from pathlib import Path

import pandas as pd

from text_cleaner import TextCleaner
from utils import parallelize_dataframe, setup_log


_logger = logging.getLogger(__name__)


def _retrieve_urls(text: str) -> str:
    regex = '\\b(ht|f)tp[s]?://[a-zA-Z0-9:/?_.&=#\-\~\˜]+[.,\\d\)]?'
    urls = []

    for match in re.finditer(regex, text):
        start, end = match.span()
        url = text[start:end].strip()
        if url.endswith('.') or url.endswith(',') or url.endswith(')'):
            url = url[:-1].strip()
        url = re.sub('\˜', '~', url)
        _logger.debug(f'Found {url}')
        urls.append(url)

    urls = list(set(urls))
    sorted_urls = []
    for url in reversed(urls):
        if 'github' in url or 'gitlab' in url:
            sorted_urls.insert(0, url)
        else:
            sorted_urls.append(url)

    return ' '.join(sorted_urls)


def _clean_and_get_urls_for_paper(paper: pd.Series, stop_when='') -> None:
    text_cleaner = TextCleaner(debug=True)
    title = paper["title"]
    text = literal_eval(paper["paper"])
    stop = len(stop_when) > 0

    _logger.debug(
        f'\nText from PDF:\n###########################\n{text}\n###########################')

    text = text_cleaner.remove_from_word_to_end(
        text, 'references')
    text = text_cleaner.remove_from_word_to_end(
        text, 'acknowledgment')
    text = text_cleaner.remove_from_word_to_end(
        text, 'acknowledgement')
    text = text_cleaner.remove_before_title(text, title)
    # text = text_cleaner.remove_between_title_abstract(text, title)
    text = text_cleaner.replace_symbol_by_letters(text)
    text = text_cleaner.remove_cid(text)
    text = text_cleaner.remove_equations(text)
    text = text_cleaner.remove_numbers_only_lines(text)
    text = text_cleaner.remove_tabular_data(text)
    # run these 3 again to remove consecutive lines
    text = text_cleaner.remove_equations(text)
    text = text_cleaner.remove_numbers_only_lines(text)
    text = text_cleaner.remove_tabular_data(text)
    text = ' '.join(text.split())
    text = text_cleaner.aglutinate_urls(text)
    text = _retrieve_urls(text)
    urls = '\n'.join(text.split())
    _logger.info(f'\n###########################\n\nTitle: \n{title}')
    _logger.info(
        f'\n###########################\n\nURLs:\n{urls}')


def _clean_and_get_urls(df: pd.DataFrame) -> pd.DataFrame:
    text_cleaner = TextCleaner()
    logger = logging.getLogger(__name__)

    # drop papers that are not usable
    total_papers = len(df)
    min_title_len = 4
    df_not_nan = df[df['title'].notna() & df['paper'].notna()]
    new_df = df_not_nan[df_not_nan['title'].map(len) > min_title_len]
    new_total_papers = len(new_df)

    if total_papers - new_total_papers > 0:
        _logger.debug(f'Dropped {total_papers - new_total_papers} papers')
        df = new_df

    df.loc[:, 'paper'] = df['paper'].apply(literal_eval)
    df.loc[:, 'paper'] = df['paper'].apply(
        text_cleaner.remove_from_word_to_end, from_word='references')
    df.loc[:, 'paper'] = df['paper'].apply(
        text_cleaner.remove_from_word_to_end, from_word='acknowledgment')
    df.loc[:, 'paper'] = df['paper'].apply(
        text_cleaner.remove_from_word_to_end, from_word='acknowledgement')
    df.loc[:, 'paper'] = df.apply(
        _remove_before_title, axis=1, text_cleaner=text_cleaner)
    # df.loc[:, 'paper'] = df.apply(
    #     _remove_between_title_abstract, axis=1, text_cleaner=text_cleaner)
    df.loc[:, 'paper'] = df['paper'].apply(
        text_cleaner.replace_symbol_by_letters)
    df.loc[:, 'paper'] = df['paper'].apply(
        text_cleaner.remove_cid)
    df.loc[:, 'paper'] = df['paper'].apply(
        text_cleaner.remove_equations)
    df.loc[:, 'paper'] = df['paper'].apply(
        text_cleaner.remove_numbers_only_lines)
    df.loc[:, 'paper'] = df['paper'].apply(
        text_cleaner.remove_tabular_data)
    # run these 3 again to remove consecutive lines
    df.loc[:, 'paper'] = df['paper'].apply(
        text_cleaner.remove_equations)
    df.loc[:, 'paper'] = df['paper'].apply(
        text_cleaner.remove_numbers_only_lines)
    df.loc[:, 'paper'] = df['paper'].apply(
        text_cleaner.remove_tabular_data)
    df.loc[:, 'paper'] = df['paper'].str.split().str.join(' ')
    df.loc[:, 'paper'] = df['paper'].apply(
        text_cleaner.aglutinate_urls)
    df.loc[:, 'paper'] = df['paper'].apply(_retrieve_urls)
    return df


def _remove_before_title(row: pd.Series, text_cleaner: TextCleaner) -> str:
    return text_cleaner.remove_before_title(row['paper'], row['title'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract URLs from papers.")
    parser.add_argument('-f', '--file', type=str, default='',
                        help='file name with paper info')
    parser.add_argument('-t', '--title', type=str, default='',
                        help='title of paper to debug')
    parser.add_argument('-i', '--index', type=int, default=-1,
                        help='index of paper to debug')
    parser.add_argument('-l', '--log_level', type=str, default='warning',
                        choices=('debug', 'info', 'warning',
                                 'error', 'critical', 'print'),
                        help='log level to debug')
    parser.add_argument('-s', '--separator', type=str, default='<#sep#>',
                        help='csv separator')
    args = parser.parse_args()

    log_dir = Path('logs/').expanduser()
    log_dir.mkdir(exist_ok=True)
    setup_log(args.log_level, log_dir / 'url_scrapper.log')

    if len(args.separator) == 1:
        df = pd.read_csv(args.file, sep=args.separator,
                         dtype=str, keep_default_na=False)
    else:
        df = pd.read_csv(args.file, sep=args.separator,
                         dtype=str, engine='python', keep_default_na=False)

    if args.index != -1:
        paper = df.iloc[args.index]
        specific_paper = True
    elif len(args.title) > 0:
        file_to_find = args.title.lower()
        found_papers = df.loc[df['title'].str.lower().str.find(
            file_to_find) >= 0]
        if len(found_papers) == 0:
            _logger.error(
                f"Couldn't find any paper with '{args.title}' in title")
            exit(0)
        elif len(found_papers) > 1:
            _logger.info(
                f'Found {len(found_papers)} papers with "{args.title}" in title. Using the first one')

        paper = found_papers.iloc[0]
        specific_paper = True

    else:
        specific_paper = False

    if specific_paper:
        _clean_and_get_urls_for_paper(paper)
    else:
        n_subprocesses = multiprocessing.cpu_count()//2
        if len(df) < n_subprocesses * 3:
            df = _clean_and_get_urls(df)
        else:
            df = parallelize_dataframe(
                df, _clean_and_get_urls, n_subprocesses)
        df = df.rename(columns={'paper': 'urls'})
        new_file_name = Path(args.file).name
        new_file_name = new_file_name.split('.')
        new_file_name = '.'.join(
            new_file_name[:-1]) + '_urls.' + new_file_name[-1]

        df.to_csv(Path(args.file).parent / new_file_name, sep='|', index=False)

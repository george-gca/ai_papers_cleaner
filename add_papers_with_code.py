import argparse
import logging
from pathlib import Path
import ftfy
import json
import multiprocessing
import re
from difflib import SequenceMatcher
from typing import Dict, Set, Union

import pandas as pd
from tqdm import tqdm

from text_cleaner import _clean_abstracts, TextCleaner
from utils import parallelize_dataframe, setup_log


_logger = logging.getLogger(__name__)


def _add_clean_title(d: Dict[str, Union[str, bool]], text_cleaner: TextCleaner) -> Dict[str, Union[str, bool]]:
    d['clean_title'] = _clean_title(d['title'], text_cleaner)
    return d


def _clean_abstract(d: Dict[str, Union[str, bool]]) -> Dict[str, Union[str, bool]]:
    text = [t for t in ftfy.fix_text(d['abstract']).split('\n') if len(t.strip()) > 0]
    text = ' '.join(text).strip()
    text = [t for t in text.split() if len(t.strip()) > 0]
    d['abstract'] = ' '.join(text).strip()
    return d


def _convert_abstract_to_repr(d: Dict[str, Union[str, bool]]) -> Dict[str, Union[str, bool]]:
    d['abstract'] = repr(d['abstract'])
    return d


def _clean_title(title: str, text_cleaner: TextCleaner) -> str:
    title = ftfy.fix_text(title)
    title = text_cleaner.remove_accents(title.lower())
    title = title.replace('\\', '')
    title = text_cleaner.remove_symbols(title).strip()
    title = title.replace('--', '-')
    title = title.replace('–', '-')
    title = title.replace('−', '-')
    return ' '.join(title.split())


def _discard_keys(d: Dict[str, Union[str, bool]], keys_to_keep: Set[str]) -> Dict[str, Union[str, bool]]:
    paper = {k: v for k, v in d.items() if k in keys_to_keep}
    return paper


def _merge_dicts(d1: Dict[str, Union[str, bool]], d2: Dict[str, Union[str, bool]]) -> Dict[str, Union[str, bool]]:
    if d1['paper_url'] in d2:
        d1 = {**d1, **d2[d1['paper_url']]}
    return d1


def _rename_keys(d: Dict[str, Union[str, bool]], regex: re.Pattern) -> Dict[str, Union[str, bool]]:
    abs_url = d['url_abs'].lower()
    if d['arxiv_id'] is not None and len(d['arxiv_id']) > 0:
        arxiv_id = d['arxiv_id']
        d['abstract_url'] = arxiv_id
        d['pdf_url'] = arxiv_id
        d['conference'] = 'arxiv'
        d.pop('url_abs')
        d.pop('url_pdf')
    elif 'content_cvpr' in abs_url:
        d['abstract_url'] = d.pop('url_abs').replace('https://openaccess.thecvf.com/', '')
        d['pdf_url'] = d.pop('url_pdf').replace('https://openaccess.thecvf.com/', '')
        d['conference'] = 'cvpr'
    elif 'content_eccv' in abs_url:
        d['abstract_url'] = d.pop('url_abs').replace('https://openaccess.thecvf.com/', '')
        d['pdf_url'] = d.pop('url_pdf').replace('https://openaccess.thecvf.com/', '')
        d['conference'] = 'eccv'
    elif 'nips.cc' in abs_url:
        d['abstract_url'] = d.pop('url_abs')
        d['pdf_url'] = d.pop('url_pdf')
        d['conference'] = 'neurips'
    elif 'aclanthology' in abs_url or 'aclweb' in abs_url:
        url_abs = d.pop('url_abs')
        d['abstract_url'] = url_abs
        url_abs = url_abs.replace('https://aclanthology.org/volumes/', '')
        url_abs = url_abs.replace('https://www.aclweb.org/anthology/', '')
        url_abs = url_abs.replace('https://aclanthology.org/', '')
        url_abs = url_abs.replace('/', '')

        d['pdf_url'] = d.pop('url_pdf')
        d['conference'] = regex.sub(r'\1', url_abs)
        if d['conference'] == 'jeptalnrecital':
            d['conference'] = 'recital'
    elif 'icml.cc' in abs_url:
        d['abstract_url'] = d.pop('url_abs')
        d['pdf_url'] = d.pop('url_pdf')
        d['conference'] = 'icml'
    else:
        d['abstract_url'] = d.pop('url_abs')
        d['pdf_url'] = d.pop('url_pdf')
        d['conference'] = 'none'

    d.pop('arxiv_id')
    d['urls'] = d.pop('paper_url')
    d['year'] = int(d.pop('date')[:4])
    return d


# TODO: split abstracts.feather into two:
# one with columns title, conference, year, clean_title
# other with columns clean_title, abstract
# save them with .to_json('.json.gz')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Add papers from paperswithcode.")
    parser.add_argument('-a', '--abstracts_file', type=str, default='data/abstracts.feather',
                        help='file name with papers abstracts')
    parser.add_argument('--codes_file', type=str, default='data/papers_with_code/links-between-papers-and-code.json',
                        help='file name with papers abstracts')
    parser.add_argument('-l', '--log_level', type=str, default='info',
                        choices=('debug', 'info', 'warning',
                                 'error', 'critical', 'print'),
                        help='log level to debug')
    parser.add_argument('--papers_file', type=str, default='data/papers_with_code/papers-with-abstracts.json',
                        help='file name with papers abstracts')
    parser.add_argument('--papers_info_file', type=str, default='data/paper_info.feather',
                        help='file name with papers infos')
    parser.add_argument('-s', '--separator', type=str, default='|',
                        help='csv separator')
    parser.add_argument('-u', '--urls_file', type=str, default='data/pdfs_urls.feather',
                        help='file name with papers urls')
    args = parser.parse_args()

    log_dir = Path('logs/').expanduser()
    log_dir.mkdir(exist_ok=True)
    setup_log(args, log_dir / 'add_papers_with_code.log')

    # loading abstracts
    abstracts_file = Path(args.abstracts_file).expanduser()
    df = pd.read_feather(abstracts_file)
    df.dropna(inplace=True)

    # loading clean abstracts
    abstracts_clean_file = abstracts_file.parent / f'{abstracts_file.stem}_clean{abstracts_file.suffix}'
    df_clean = pd.read_feather(abstracts_clean_file)
    df_clean.dropna(inplace=True)

    # loading urls
    urls_file = Path(args.urls_file).expanduser()
    df_urls = pd.read_feather(urls_file)
    df_urls.dropna(inplace=True)

    # loading infos
    papers_info_file = Path(args.papers_info_file).expanduser()
    df_infos = pd.read_feather(papers_info_file)
    df_infos.dropna(inplace=True)

    # loading abstracts from papers with code
    papers_file = Path(args.papers_file).expanduser()
    with open(papers_file) as f:
        papers_abstracts = json.load(f)

    useful_keys = {
        'abstract',
        'arxiv_id',
        'date',
        'paper_url',
        'title',
        'url_abs',
        'url_pdf',
    }

    papers_abstracts = [_discard_keys(d, useful_keys) for d in papers_abstracts]

    # loading codes from papers with code
    with open(args.codes_file) as f:
        papers_codes = json.load(f)

    useful_keys = {
        'paper_url',
        'repo_url',
    }

    papers_codes = [_discard_keys(d, useful_keys) for d in papers_codes]

    # merge info from papers with code
    # discarding papers with title longer than 230 characters and without year
    papers_codes = {d['paper_url']: d for d in papers_codes}
    papers_abstracts = [_merge_dicts(d, papers_codes) for d in papers_abstracts if d['title'] and 0 < len(d['title']) < 230 and len(d['date']) > 0]

    # check if this paper is not already in the abstracts df
    # clean and compare the titles
    text_cleaner = TextCleaner()
    df['clean_title'] = df['title'].str.lower()
    df.loc[:, 'clean_title'] = df['clean_title'].apply(
        ftfy.fix_text)
    df.loc[:, 'clean_title'] = df['clean_title'].apply(
        text_cleaner.remove_accents)
    df.loc[:, 'clean_title'] = df['clean_title'].str.replace('\\', '')
    df.loc[:, 'clean_title'] = df['clean_title'].apply(
        text_cleaner.remove_symbols)
    df.loc[:, 'clean_title'] = df['clean_title'].str.replace('--', '-')
    df.loc[:, 'clean_title'] = df['clean_title'].str.replace('–', '-')
    df.loc[:, 'clean_title'] = df['clean_title'].str.replace('−', '-')
    df.loc[:, 'clean_title'] = df['clean_title'].str.strip().str.split().str.join(' ')

    df_urls['clean_title'] = df_urls['title'].str.lower()
    df_urls.loc[:, 'clean_title'] = df_urls['clean_title'].apply(
        ftfy.fix_text)
    df_urls.loc[:, 'clean_title'] = df_urls['clean_title'].apply(
        text_cleaner.remove_accents)
    df_urls.loc[:, 'clean_title'] = df_urls['clean_title'].str.replace('\\', '')
    df_urls.loc[:, 'clean_title'] = df_urls['clean_title'].apply(
        text_cleaner.remove_symbols)
    df_urls.loc[:, 'clean_title'] = df_urls['clean_title'].str.replace('--', '-')
    df_urls.loc[:, 'clean_title'] = df_urls['clean_title'].str.replace('–', '-')
    df_urls.loc[:, 'clean_title'] = df_urls['clean_title'].str.replace('−', '-')
    df_urls.loc[:, 'clean_title'] = df_urls['clean_title'].str.strip().str.split().str.join(' ')

    papers = {_clean_title(d['title'], text_cleaner): d for d in papers_abstracts if d['title'] is not None and d['abstract'] is not None}

    # if it is, just add the new url to the urls dataframe
    # df_urls[df_urls.clean_title.isin(papers)] = df_urls[df_urls.clean_title.isin(papers)].apply(_add_urls, papers_urls=papers, axis=1)

    # if not, add it to all dataframes
    papers_already_in = {s for s in df[df.clean_title.isin(papers)].clean_title}
    _logger.info(f'\n{len(papers_already_in):n} papers with info already found')

    # also consider papers with similar titles as already in previous data
    papers_not_in = {k: v for k, v in papers.items() if k not in papers_already_in}
    seq_matcher = SequenceMatcher()
    similar_titles = set()
    _logger.info('\nPrinting similar titles')
    for k, v in tqdm(papers_not_in.items(), desc='Similar titles'):
        seq_matcher.set_seq2(v['title'])
        for _, t in df.title[abs(df.title.str.len() - len(v['title'])) < 5].iteritems():
            seq_matcher.set_seq1(t)

            if seq_matcher.real_quick_ratio() > 0.95 and seq_matcher.quick_ratio() > 0.95:  # and seq_matcher.ratio() > 0.95:
                similar_titles.add(k)
                tqdm.write(f'{t}\n{v["title"]}\n')
                break

    _logger.info(f'\n{len(similar_titles):n} similar titles')

    papers_already_in = papers_already_in.union(similar_titles)
    papers_not_in = [v for k, v in papers.items() if k not in papers_already_in]
    _logger.info(f'\n{len(papers_not_in):n} new papers')

    # rename dicts' keys, filter by years and clean
    acl_conference_regex = re.compile(r'[\d]+.([\w]+)-[\d\w.]+')
    papers_not_in = [_rename_keys(d, acl_conference_regex) for d in papers_not_in]
    papers_not_in = [d for d in papers_not_in if d['year'] >= 2017 and not 'openreview.net' in d['abstract_url']]
    papers_not_in = [_clean_abstract(d) for d in papers_not_in]
    papers_not_in = [_add_clean_title(d, text_cleaner) for d in papers_not_in]

    _logger.info(f'\n{len(papers_not_in):n} papers to be added')

    _logger.info('\nPrinting some data:')
    for i in range(5):
        _logger.info(papers_not_in[i])

    # creating new abstracts with added papers_with_code info
    useful_keys = {
        'abstract',
        'clean_title',
        'conference',
        'title',
        'year',
    }
    add_to_abstracts = [_discard_keys(d, useful_keys) for d in papers_not_in]
    _logger.info(f'\nWe had abstracts for {len(df):n} papers')
    df = df.append(add_to_abstracts, ignore_index=True)
    _logger.info(f'Now we have abstracts for {len(df):n} papers')
    # df.to_csv(abstracts_file.parent / 'abstracts_pwc.csv', sep='|', index=False)
    df['year'] = df['year'].astype('int')
    df.to_feather(abstracts_file.parent / 'abstracts_pwc.feather', compression='zstd')

    # creating abstracts.csv inside papers_with_code dir
    useful_keys = {
        'abstract',
        'conference',
        'title',
        'year',
    }
    create_abstracts = [_discard_keys(d, useful_keys) for d in papers_not_in]
    create_abstracts = [_convert_abstract_to_repr(d) for d in create_abstracts]
    df_abstracts = pd.DataFrame(create_abstracts)
    df_abstracts.to_csv(papers_file.parent / 'abstracts.csv', sep='|', index=False)

    # creating new paper_info with added papers_with_code info
    useful_keys = {
        'abstract_url',
        'conference',
        'pdf_url',
        'title',
        'year',
    }
    add_to_paper_info = [_discard_keys(d, useful_keys) for d in papers_not_in]
    _logger.info(f'\nWe had info for {len(df_infos):n} papers')
    df_infos = df_infos.append(add_to_paper_info, ignore_index=True)
    _logger.info(f'Now we have info for {len(df_infos):n} papers')
    # df_infos.to_csv(papers_info_file.parent / 'paper_info_pwc.csv', sep=';', index=False)
    df_infos['year'] = df_infos['year'].astype('int')
    df_infos.to_feather(papers_info_file.parent / 'paper_info_pwc.feather', compression='zstd')

    # creating paper_info.csv inside papers_with_code dir
    useful_keys = {
        'abstract_url',
        'pdf_url',
        'title',
    }
    create_paper_info = [_discard_keys(d, useful_keys) for d in papers_not_in]
    df_paper_info = pd.DataFrame(create_paper_info)
    df_paper_info.to_csv(papers_file.parent / 'paper_info.csv', sep=';', index=False)

    # creating new pdfs_urls with added papers_with_code info
    useful_keys = {
        'clean_title',
        'title',
        'urls',
    }
    add_to_pdfs_urls = [_discard_keys(d, useful_keys) for d in papers_not_in]
    _logger.info(f'\nWe had urls for {len(df_urls):n} papers')
    df_urls = df_urls.append(add_to_pdfs_urls, ignore_index=True)
    _logger.info(f'Now we have urls for {len(df_urls):n} papers')
    # df_urls.to_csv(urls_file.parent / 'pdfs_urls_pwc.csv', sep='|', index=False)
    df_urls.to_feather(urls_file.parent / 'pdfs_urls_pwc.feather', compression='zstd')

    # creating pdfs_urls.csv inside papers_with_code dir
    useful_keys = {
        'title',
        'urls',
    }
    create_pdfs_urls = [_discard_keys(d, useful_keys) for d in papers_not_in]
    df_pdfs_urls = pd.DataFrame(create_pdfs_urls)
    df_pdfs_urls.to_csv(papers_file.parent / 'pdfs_urls.csv', sep='|', index=False)
    # df_pdfs_urls.to_feather(papers_file.parent / 'pdfs_urls.feather', compression='zstd')

    # cleaning abstracts
    n_processes = 3*multiprocessing.cpu_count()//4
    df_abstracts_clean = parallelize_dataframe(df_abstracts, _clean_abstracts, n_processes)
    df_abstracts_clean.to_csv(papers_file.parent / 'abstracts_clean.csv', sep='|', index=False)

    # creating new abstracts_clean with added papers_with_code info
    df_clean = pd.concat([df_clean, df_abstracts_clean], ignore_index=True)
    # df_clean.to_csv(abstracts_file.parent / 'abstracts_clean_pwc.csv', sep='|', index=False)
    df_clean['year'] = df_clean['year'].astype('int')
    df_clean.to_feather(abstracts_file.parent / 'abstracts_clean_pwc.feather', compression='zstd')
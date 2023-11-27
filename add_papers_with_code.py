import argparse
import logging
from pathlib import Path
from typing import Any
import ftfy
import json
import re
from difflib import SequenceMatcher
from multiprocessing import cpu_count

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from text_cleaner import _clean_abstracts, _clean_titles, TextCleaner
from utils import parallelize_dataframe, setup_log


_logger = logging.getLogger(__name__)
KEEP_ARXIV_ID = True


def _add_clean_title(d: dict[str, str | bool], text_cleaner: TextCleaner) -> dict[str, str | bool]:
    d['clean_title'] = _clean_title(d['title'], text_cleaner)
    return d


def _clean_abstract(d: dict[str, str | bool]) -> dict[str, str | bool]:
    text = [t for t in ftfy.fix_text(d['abstract']).split('\n') if len(t.strip()) > 0]
    text = ' '.join(text).strip()
    text = [t for t in text.split() if len(t.strip()) > 0]
    d['abstract'] = ' '.join(text).strip()
    return d


def _convert_abstract_to_repr(d: dict[str, str | bool]) -> dict[str, str | bool]:
    d['abstract'] = repr(d['abstract'])
    return d


def _clean_title(title: str, text_cleaner: TextCleaner) -> str:
    clean_title = ftfy.fix_text(title.lower())
    clean_title = text_cleaner.remove_accents(clean_title)
    clean_title = text_cleaner.remove_latex_commands(clean_title)
    clean_title = text_cleaner.remove_latex_inline_equations(clean_title)
    clean_title = clean_title.replace('\\', '')
    clean_title = text_cleaner.remove_symbols(clean_title)
    clean_title = clean_title.replace('--', '-')
    clean_title = clean_title.replace('–', '-')
    clean_title = clean_title.replace('−', '-')
    clean_title = ' '.join(clean_title.strip().split())
    clean_title = text_cleaner.remove_hyphens_slashes(clean_title)
    clean_title = text_cleaner.remove_stopwords(clean_title)
    clean_title = text_cleaner.plural_to_singular(clean_title)
    clean_title = text_cleaner.replace_hyphens_by_underline(clean_title)

    return clean_title


def _discard_keys(d: dict[str, str | bool], keys_to_keep: set[str]) -> dict[str, str | bool]:
    paper = {k: v for k, v in d.items() if k in keys_to_keep}
    return paper


def _merge_dicts(d1: dict[str, str | bool], d2: dict[str, str | bool]) -> dict[str, str | bool]:
    if d1['paper_url'] in d2:
        d1 = {**d1, **d2[d1['paper_url']]}
    return d1


def _rename_keys(d: dict[str, str | bool], regex: re.Pattern) -> dict[str, str | bool]:
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

    if not KEEP_ARXIV_ID:
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
    parser.add_argument('-p', '--run_parallel', action='store_true',
                        help='run the code in multiple processes')
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
    setup_log(args.log_level, log_dir / 'add_papers_with_code.log')

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
    # FIXME this is emptying the df_urls
    df_urls.dropna(inplace=True)

    # loading infos
    papers_info_file = Path(args.papers_info_file).expanduser()
    df_infos = pd.read_feather(papers_info_file)
    df_infos.dropna(inplace=True)

    if not (len(df) == len(df_clean) == len(df_infos)):
        _logger.error(f'DataFrames must have the same length, but instead are {len(df)}, {len(df_clean)}, and {len(df_infos)}')
        raise ValueError(f'DataFrames must have the same length, but instead are {len(df)}, {len(df_clean)}, and {len(df_infos)}')

    _logger.info(f'Loaded {len(df):n} abstracts from {abstracts_file}')
    _logger.info(f'Loaded {len(df_clean):n} clean abstracts from {abstracts_clean_file}')
    _logger.info(f'Loaded {len(df_urls):n} urls from {urls_file}')
    _logger.info(f'Loaded {len(df_infos):n} infos from {papers_info_file}')

    # loading abstracts from papers with code
    papers_file = Path(args.papers_file).expanduser()
    with open(papers_file) as f:
        papers_abstracts = json.load(f)

    _logger.info(f'Loaded {len(papers_abstracts):n} papers from papers with code')

    useful_keys = {
        'abstract',
        'date',
        'paper_url',
        'title',
        'url_abs',
        'url_pdf',
    }

    if KEEP_ARXIV_ID:
        useful_keys.add('arxiv_id')

    papers_abstracts = [_discard_keys(d, useful_keys) for d in papers_abstracts]

    # loading codes from papers with code
    with open(args.codes_file) as f:
        papers_from_pwc = json.load(f)

    _logger.info(f'Loaded {len(papers_from_pwc):n} papers codes from papers with code')

    useful_keys = {
        'paper_url',
        'repo_url',
    }

    papers_from_pwc = [_discard_keys(d, useful_keys) for d in papers_from_pwc]

    # merge info from papers with code
    # discarding papers with title longer than 230 characters and without year
    papers_from_pwc = {d['paper_url']: d for d in papers_from_pwc}
    papers_abstracts = [_merge_dicts(d, papers_from_pwc) \
                        for d in papers_abstracts \
                            if d['title'] and 0 < len(d['title']) < 230 and len(d['date']) > 0]
    _logger.info(f'After merging, {len(papers_abstracts):n} papers remain from papers with code')

    # check if this paper is not already in the abstracts df
    # clean and compare the titles
    df = _clean_titles(df, progress=True)
    df_urls = _clean_titles(df_urls, progress=True)

    text_cleaner = TextCleaner()
    papers = {_clean_title(d['title'], text_cleaner): d \
              for d in tqdm(papers_abstracts, unit='paper', ncols=250, desc='Cleaning PWC titles') \
                if d['title'] is not None and d['abstract'] is not None}

    # if it is, just add the new url to the urls dataframe
    # df_urls[df_urls.clean_title.isin(papers)] = \
    #     df_urls[df_urls.clean_title.isin(papers)].apply(_add_urls, papers_urls=papers, axis=1)

    # if not, add it to all dataframes
    papers_already_in = {s for s in df[df.clean_title.isin(papers)].clean_title}

    if KEEP_ARXIV_ID:
        papers_already_in_dict = {k: v for k, v in papers.items() if k in papers_already_in}

    _logger.info(f'\n{len(papers_already_in):n} papers with info already found')

    # also consider papers with similar titles as already in previous data
    papers_not_in = {k: v for k, v in papers.items() if k not in papers_already_in}

    if args.run_parallel:
        def _find_similar_titles(papers_to_check: dict[str, Any]) -> dict[str, str]:
            seq_matcher = SequenceMatcher()
            cleaner = TextCleaner()
            similar_titles = {}
            for k, v in papers_to_check:
                seq_matcher.set_seq2(v['title'])
                # TODO: change this for for loop with iterrows
                for _, t in df.title[abs(df.title.str.len() - len(v['title'])) < 5].items():
                    seq_matcher.set_seq1(t)

                    if seq_matcher.real_quick_ratio() > 0.95 and seq_matcher.quick_ratio() > 0.95:
                        # could also add: and seq_matcher.ratio() > 0.95:
                        # TODO: use clean_title already created instead of recreating it here
                        similar_titles[_clean_title(t, cleaner)] = k
                        break

            return similar_titles

        n_processes = 3 * cpu_count() // 4
        chunk_size = len(papers_not_in) // n_processes
        list_papers_not_in = list(papers_not_in.items())
        list_chunked = [list_papers_not_in[i:i + chunk_size] for i in range(0, len(list_papers_not_in), chunk_size)]
        results: list[dict[str, str]] = process_map(_find_similar_titles, list_chunked, max_workers=n_processes)
        similar_titles_dict = {k: v for s in results for k, v in s.items()}
        similar_titles = set(similar_titles_dict.values())

    else:
        seq_matcher = SequenceMatcher()
        similar_titles = set()
        similar_titles_dict = {}
        _logger.info('\nPrinting similar titles')

        for k, v in tqdm(papers_not_in.items(), desc='Similar titles', ncols=250):
            seq_matcher.set_seq2(v['title'])

            for _, t in df.title[abs(df.title.str.len() - len(v['title'])) < 5].iteritems():
                seq_matcher.set_seq1(t)

                if seq_matcher.real_quick_ratio() > 0.95 and seq_matcher.quick_ratio() > 0.95:
                    # could also add: and seq_matcher.ratio() > 0.95:
                    similar_titles.add(k)
                    similar_titles_dict[_clean_title(t, text_cleaner)] = k
                    tqdm.write(f'{t}\n{v["title"]}\n')
                    break

    _logger.info(f'\n{len(similar_titles):n} similar titles')

    papers_already_in = papers_already_in.union(similar_titles)
    papers_not_in = [v for k, v in papers.items() if k not in papers_already_in]
    _logger.info(f'\n{len(papers_not_in):n} new papers')

    # rename dicts' keys, filter by years and clean
    acl_conference_regex = re.compile(r'[\d]+.([\w]+)-[\d\w.]+')
    papers_not_in = [_rename_keys(d, acl_conference_regex) for d in papers_not_in]
    papers_not_in = [d for d in papers_not_in if d['year'] >= 2017 and 'openreview.net' not in d['abstract_url']]
    papers_not_in = [_clean_abstract(d) for d in papers_not_in]
    papers_not_in = [_add_clean_title(d, text_cleaner) for d in papers_not_in]

    _logger.info(f'\n{len(papers_not_in):n} papers to be added')

    _logger.info('\nPrinting some data:')
    for i in range(5):
        _logger.info(papers_not_in[i])

    # creating new paper_info with added papers_with_code info
    useful_keys = {
        'abstract_url',
        'conference',
        'pdf_url',
        'title',
        'year',
    }

    if KEEP_ARXIV_ID:
        useful_keys.add('arxiv_id')
    
    add_to_paper_info = [_discard_keys(d, useful_keys) for d in papers_not_in]
    _logger.info(f'\nWe had info for {len(df_infos):n} papers')

    if KEEP_ARXIV_ID:
        # print('Papers with arxiv_id:', sum(1 for t in df['clean_title'] if t in papers_already_in_dict or (t in similar_titles_dict and similar_titles_dict[t] in papers_already_in_dict)))
        # print('From:', len(df['clean_title']))

        df_infos['arxiv_id'] = [ \
            papers_already_in_dict[t]['arxiv_id'] if t in papers_already_in_dict else
            papers_already_in_dict[similar_titles_dict[t]]['arxiv_id'] if t in similar_titles_dict and similar_titles_dict[t] in papers_already_in_dict else None
            for t in df['clean_title']
        ]

    df_infos = pd.concat([df_infos, pd.DataFrame(add_to_paper_info)], ignore_index=True)
    _logger.info(f'Now we have info for {len(df_infos):n} papers')
    df_infos['year'] = df_infos['year'].astype('int')
    df_infos.to_feather(papers_info_file.parent / 'paper_info_pwc.feather', compression='zstd')

    # creating paper_info.csv inside papers_with_code dir
    useful_keys = {
        'abstract_url',
        'pdf_url',
        'title',
    }

    if KEEP_ARXIV_ID:
        useful_keys.add('arxiv_id')
    
    create_paper_info = [_discard_keys(d, useful_keys) for d in papers_not_in]
    df_paper_info = pd.DataFrame(create_paper_info)
    df_paper_info.to_csv(papers_file.parent / 'paper_info.csv', sep=';', index=False)

    # creating new abstracts with added papers_with_code info
    useful_keys = {
        'abstract',
        'conference',
        'title',
        'year',
    }
    add_to_abstracts = [_discard_keys(d, useful_keys) for d in papers_not_in]
    _logger.info(f'\nWe had abstracts for {len(df):n} papers')
    df = pd.concat([df, pd.DataFrame(add_to_abstracts)], ignore_index=True)
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

    # creating new pdfs_urls with added papers_with_code info
    useful_keys = {
        'clean_title',
        'title',
        'urls',
    }
    add_to_pdfs_urls = [_discard_keys(d, useful_keys) for d in papers_not_in]
    _logger.info(f'\nWe had urls for {len(df_urls):n} papers')
    df_urls = pd.concat([df_urls, pd.DataFrame(add_to_pdfs_urls)], ignore_index=True)
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
    n_processes = 3*cpu_count()//4
    df_abstracts_clean = parallelize_dataframe(df_abstracts, _clean_abstracts, n_processes)
    df_abstracts_clean = parallelize_dataframe(df_abstracts_clean, _clean_titles, n_processes)
    df_abstracts_clean.to_csv(papers_file.parent / 'abstracts_clean.csv', sep='|', index=False)

    # creating new abstracts_clean with added papers_with_code info
    _logger.info(f'\nWe had clean abstracts for {len(df_clean):n} papers')
    df_clean = pd.concat([df_clean, df_abstracts_clean], ignore_index=True)
    _logger.info(f'Now we have clean abstracts for {len(df_clean):n} papers')
    # df_clean.to_csv(abstracts_file.parent / 'abstracts_clean_pwc.csv', sep='|', index=False)
    df_clean['year'] = df_clean['year'].astype('int')
    df_clean.to_feather(abstracts_file.parent / 'abstracts_clean_pwc.feather', compression='zstd')

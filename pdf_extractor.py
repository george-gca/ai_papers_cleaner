import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
# https://github.com/py-pdf/benchmarks
import pypdfium2 as pdfium
from tqdm import tqdm

from utils import setup_log


_logger = logging.getLogger(__name__)


def extract_pdf_name(pdf_url: str, dirname: str, pdf_suffix: str) -> str:
    pdf_name = pdf_url.split('/')[-1]
    if pdf_name[-4:] == '.pdf':
        return str(Path(dirname) / pdf_name)
    else:
        return str(Path(dirname) / f'{pdf_name.split("/")[-1]}{pdf_suffix}.pdf')


def extract_text(filepath: str) -> str:
    text = ""
    with open(filepath, "rb") as f:
        data = f.read()

    pdf = pdfium.PdfDocument(data)
    for i in range(len(pdf)):
        page = pdf.get_page(i)
        text_page = page.get_textpage()
        text += text_page.get_text()
        text += "\n"
        [g.close() for g in (text_page, page)]
    pdf.close()
    return text


def extract_title(page: str, conference: str, year: str) -> str:
    lines = page.split('\n', 20)
    begin = -1
    end = 0
    for i, line in enumerate(lines):
        clean_line = line.strip()
        if begin == -1 and len(clean_line) > 0 and \
           not conference in clean_line.lower() and not year in clean_line:
            begin = i
        elif begin != -1 and (len(clean_line) == 0 or len(clean_line) > 50):
            end = i
            break

    title = ' '.join([w.strip() for w in lines[begin:end]]).strip()
    if title.upper() == title:
        title = title.title()

    if title.isnumeric():
        return ''

    return title


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract papers' titles and abstracts.")
    parser.add_argument('-c', '--conference', type=str, required=True,
                        help='conference to scrape data')
    parser.add_argument('-f', '--file', type=str, default='',
                        help='part of file name to debug')
    parser.add_argument('-i', '--index', type=int, default=-1,
                        help='file index to debug')
    parser.add_argument('-l', '--log_level', type=str, default='warning',
                        choices=('debug', 'info', 'warning',
                                 'error', 'critical', 'print'),
                        help='log level to debug')
    parser.add_argument('-p', '--use_paper_info', action='store_true',
                        help='paper infos file')
    parser.add_argument('--pdf_suffix', type=str, default='',
                        help='pdf suffix')
    parser.add_argument('-s', '--separator', type=str, default='<#sep#>',
                        help='csv separator')
    parser.add_argument('-y', '--year', type=str, required=True,
                        help='year of the conference')
    args = parser.parse_args()

    log_dir = Path('logs/').expanduser()
    log_dir.mkdir(exist_ok=True)
    setup_log(args, log_dir / 'pdf_extractor.log')

    conference_year_dir = Path('data') / args.conference / args.year
    papers = conference_year_dir / 'papers'
    papers = [str(p) for p in papers.glob('*.pdf')]

    if args.index != -1:
        paper = papers[args.index]
    elif len(args.file) > 0:
        file_to_find = args.file.lower()
        found_papers = [
            s for s in papers if file_to_find in Path(s).name.lower()]
        if len(found_papers) == 0:
            _logger.error(
                f"Couldn't find any paper with {args.file} in file name")
            sys.exit(0)
        elif len(found_papers) > 1:
            _logger.info(
                f'Found {len(found_papers)} papers with "{args.file}" in name. Using the first one: {found_papers[0]}')

        paper = found_papers[0]
    else:
        paper = ''

    if len(paper) > 0:
        text = extract_text(paper)
        title = extract_title(text, args.conference, args.year)

        _logger.debug(
            f'\nText from PDF:\n###########################\n{text}\n###########################')

        _logger.info(f'\nPaper:\n{paper}')
        _logger.info(f'\nTitle: \n{title}')
        # _logger.info(f'\nText: \n{repr(text)}')

    else:
        if args.use_paper_info:
            paper_info = conference_year_dir / 'paper_info.csv'

            if paper_info.exists():
                paper_info_df = pd.read_csv(paper_info, sep=';')
                paper_info_df.loc[:, 'pdf_url'] = paper_info_df['pdf_url'].astype(str)
                paper_info_df.loc[:, 'pdf_url'] = paper_info_df['pdf_url'].apply(
                    extract_pdf_name, dirname=str(Path(papers[0]).parent), pdf_suffix=args.pdf_suffix)
                paper_info_df = paper_info_df.drop(columns=['abstract_url'])
                paper_info_df = paper_info_df[paper_info_df.pdf_url.isin(set(papers))]
                assert len(paper_info_df) > 0

            else:
                _logger.warning('Paper info file does not exists. Doing without it')
                args.use_paper_info = False

        with open(conference_year_dir / 'pdfs.csv', 'w') as corpus_file:
            with open(conference_year_dir / 'errors.txt', 'w') as error_file:

                corpus_file.write(f'title{args.separator}paper\n')

                if args.use_paper_info:
                    pbar = tqdm(paper_info_df.iterrows(), total=len(paper_info_df), unit='paper')

                    for index, row in pbar:
                        pbar.set_description(Path(row['pdf_url']).name[:50])

                        try:
                            text = extract_text(row['pdf_url'])
                            title = row['title']
                            corpus_file.write(
                                f'{title}{args.separator}{repr(text)}\n')

                        except:
                            _logger.error(
                                f'Error while extracting text of {Path(row["pdf_url"]).name}')
                            error_file.write(
                                f'Error while extracting text of {Path(row["pdf_url"]).name}\n')

                else:
                    pbar = tqdm(papers, total=len(papers), unit='paper')

                    for paper in pbar:
                        pbar.set_description(Path(paper).name[:50])

                        try:
                            text = extract_text(paper)
                            title = extract_title(text, args.conference, args.year)
                            corpus_file.write(
                                f'{title}{args.separator}{repr(text)}\n')

                        except:
                            _logger.error(
                                f'Error while extracting text of {Path(paper).name}')
                            error_file.write(
                                f'Error while extracting text of {Path(paper).name}\n')

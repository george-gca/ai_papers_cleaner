# Copilot Instructions for ai_papers_cleaner

## Repository Summary

`ai_papers_cleaner` is a Python CLI tool that extracts text from AI research paper PDFs and abstracts, then removes uninformative words (stop words, equations, URLs, tabular data, etc.) to produce a clean corpus suitable for training language models. It is part of a larger pipeline alongside `ai_papers_scrapper` and `ai_papers_search_tool`.

## Language & Runtime

- **Python 3.11+** (Dockerfile uses `python:3.11`; `pyproject.toml` requires `^3.10`)
- **Dependency manager**: [Poetry](https://python-poetry.org/) — always run `poetry install` before running any script locally
- **Linter**: [Ruff](https://docs.astral.sh/ruff/) — configured in `.ruff.toml` (line-length = 125)
- **No test suite** — there are no automated tests in this repository

## Repository Layout

```
/                         ← repo root
├── .ruff.toml            ← Ruff linter config (line-length = 125)
├── .gitignore            ← ignores data/, logs/, __pycache__, venvs, etc.
├── Dockerfile            ← Python 3.11 image with all pip deps installed
├── Makefile              ← Docker build/run helpers (see below)
├── pyproject.toml        ← Poetry deps: colorama, ftfy, ipython, nltk, pandas,
│                            pyenchant, pypdfium2, tqdm, inflect (git fork), pyarrow
├── poetry.lock           ← locked deps
├── start_here.sh         ← main entry point; loops over conferences/years and
│                            calls pdf_extractor.py, url_scrapper.py, text_cleaner.py
├── clean_paper.sh        ← debug helper: runs text_cleaner.py in debug mode for one paper
├── text_cleaner.py       ← core logic: TextCleaner class (stop words, lemmatization,
│                            regex cleaning, PDF text normalization)
├── pdf_extractor.py      ← extracts raw text from PDF files using pypdfium2
├── url_scrapper.py       ← extracts URLs from cleaned paper text
├── unify_papers_data.py  ← merges per-conference TSV files into unified datasets
├── add_papers_with_code.py ← enriches data with Papers With Code metadata
├── utils.py              ← shared helpers: parallelize_dataframe(), setup_log(),
│                            SUPPORTED_CONFERENCES list
└── timer.py              ← Timer context-manager/decorator utility
```

**Data layout** (not committed; lives in `~/datasets/ai_papers/` by convention):
```
data/<conf>/<year>/
    abstracts.tsv       ← tab-separated: title, abstract (input)
    abstracts_clean.tsv ← cleaned abstracts (output)
    pdfs.csv            ← CSV with custom separator <#sep#>: title, paper (input)
    paper_info.tsv      ← tab-separated metadata
    pdfs_urls.tsv       ← extracted URLs (output of url_scrapper.py)
```

## Setting Up Locally (without Docker)

```bash
# 1. Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# 2. Install dependencies (always do this first)
poetry install

# 3. Download required NLTK data (one-time setup, required at runtime)
poetry run python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# 4. Run the pipeline
bash start_here.sh
```

## Linting

Always lint after making changes:

```bash
# Install ruff if not available
pip install ruff

# Lint all Python files
ruff check *.py

# Auto-fix fixable issues
ruff check --fix *.py
```

The linter is configured in `.ruff.toml` with `line-length = 125`. There are no pre-commit hooks.

## Running with Docker

```bash
make          # builds Docker image (same as: make build)
make run      # runs start_here.sh inside the container
make start    # opens an interactive shell in the container
```

Override the command: `make RUN_STRING="bash clean_paper.sh" run`

The `Makefile` mounts:
- `$(PWD)` → `/work` (source code)
- `~/datasets/ai_papers` → `/work/data`
- `~/nltk_data` → `/home/<user>/nltk_data`

## Key Architecture Notes

- **`TextCleaner`** (`text_cleaner.py`): The main class. It uses `enchant` (spell checking), `inflect` (pluralization/singularization), `nltk` stopwords, and many regex patterns to clean text. The `inflect` dependency uses a **custom git fork**: `git+https://github.com/george-gca/inflect`.
- **`parallelize_dataframe()`** (`utils.py`): Used to process large DataFrames across multiple CPU cores via `tqdm.contrib.concurrent.process_map`.
- **`setup_log()`** (`utils.py`): Standard logging setup used by all entry-point scripts; writes to a `logs/` directory.
- **`SUPPORTED_CONFERENCES`** (`utils.py`): The canonical list of supported conference short names. Update this list when adding new conferences.
- All scripts accept `--log_level` (`debug`, `info`, `warning`, `error`, `critical`, `print`) and write logs to `logs/<script_name>.log`.
- Data files use **tab (`\t`) as the separator** for `.tsv` files and a custom multi-char separator `<#sep#>` for `pdfs.csv` (to handle text containing commas and semicolons).

## No CI/CD Pipelines

There are no GitHub Actions workflows in this repository. Validation is done manually by running `ruff check *.py` and testing scripts against real data.

# AI Papers Cleaner

Extract text from papers PDFs and abstracts, and remove uninformative words. This is helpful for building a corpus of papers to train a language model.

## Requirements

[Docker](https://www.docker.com/) or, for local installation:

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/)

## Usage

To make it easier to run the code, with or without Docker, I created a few helpers. Both ways use `start_here.sh` as an entry point. Since there are a few quirks when calling the specific code, I created this file with all the necessary commands to run the code. All you need to do is to uncomment the relevant lines inside the `conferences` array and run the script. Also, comment/uncomment the following as needed:

```bash
extract_pdfs=1
extract_urls=1
clean_abstracts=1
clean_papers=1
```

You'll need to download some nltk data. To do this, read the relevant section according to your usage method below.

### Running without Docker

You first need to install [Python Poetry](https://python-poetry.org/docs/). Then, you can install the dependencies and run the code:

```bash
poetry install
bash start_here.sh
```

#### Downloading nltk data

To download the nltk data, run the following:

```bash
poetry run ipython3
```

Then, inside the Python shell:

```python
import nltk
nltk.download('stopwords')
```

### Running with Docker

To help with the Docker setup, I created a `Dockerfile` and a `Makefile`. The `Dockerfile` contains all the instructions to create the Docker image. The `Makefile` contains the commands to build the image, run the container, and run the code inside the container. To build the image, simply run:

```bash
make
```

To call `start_here.sh` inside the container, run:

```bash
make run
```

#### Downloading nltk data

To download the nltk data, run the following:

```bash
make RUN_STRING="ipython3" run
```

Then, inside the Python shell:

```python
import nltk
nltk.download('stopwords')
```

## Checking the cleaning process

The best way to check how the cleaning process works for a specific paper is by running the [clean_paper.sh](clean_paper.sh) script. You can set inside the following variables:

```bash
# clean_abstracts=1
clean_papers=1

index=1
# title="Moon IME: Neural-based Chinese Pinyin Aided Input Method with Customizable Association"
conf=aaai
year=2017
```

To check the abstract cleaning process, uncomment the `clean_abstracts` line and comment the `clean_papers` line. To check the paper cleaning process, reverse the comments. You need to set the `conf` and `year` variables to the conference (as displayed in the `conferences` array in [start_here.sh](start_here.sh)) and year of your choice, and set one of `index` or `title` variables. The `index` variable is the index of the paper in the `abstracts.csv` or `pdfs.csv` file, while `title` can be a part of the title of the paper. If you set both, the `index` variable will be used. To call the clean_paper.sh script, run:

```bash
bash clean_paper.sh # if you're running without Docker
make RUN_STRING="bash clean_paper.sh" run # if you're running with Docker
```

# IR System 2026

An information retrieval (IR) prototype built in Python. The project focuses on:

- loading a dataset with `ir_datasets`
- preprocessing documents and queries
- building an inverted index
- validating the index with a simple test

The default dataset is `msmarco-passage/train`.

## Features

- document loading from `ir_datasets`
- text normalization, tokenization, stopword removal, and lemmatization/stemming
- inverted index construction with document frequencies and document lengths
- query preprocessing that reuses the same pipeline as documents
- reproducible processed outputs saved under `data/`

## Project Structure

- `services/config.py` - shared configuration values
- `services/preprocessing/` - dataset loading and text preprocessing
- `services/indexing/` - inverted index implementation
- `services/query_processing/` - query preprocessing
- `scripts/process_docs.py` - main script to load and preprocess documents
- `tests/test_index.py` - basic verification that the index can answer queries
- `data/` - raw, processed, and indexed outputs

## Requirements

- Python 3.11 or newer is recommended
- See `requirements.txt` for the full dependency list

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Some packages may need extra setup on first use:

- `ir_datasets` downloads the dataset automatically when you run the loader
- NLTK resources are downloaded automatically by the preprocessor if missing
- `spacy` can use `en_core_web_sm` if it is installed, but the pipeline still works without it

## Prepare Data

Run the preprocessing pipeline to load raw documents and generate processed outputs:

```powershell
python scripts\process_docs.py
```

By default, this will:

- load documents from `msmarco-passage/train`
- save raw documents to `data/raw/raw_docs.csv`
- save processed documents to:
  - `data/processed/processed_docs.pkl`
  - `data/processed/processed_docs_10k.pkl`
  - `data/processed/processed_docs.csv`
  - `data/processed/processed_docs_10k.csv`

You can override the dataset or output locations:

```powershell
python scripts\process_docs.py --dataset msmarco-passage/train --max-docs 10000 --raw-dir data/raw --processed-dir data/processed
```

## Run Tests

The current test expects processed documents to already exist in `data/processed`:

```powershell
pytest -q
```

If your processed files are stored somewhere else, set `IR_PROCESSED_DIR` first:

```powershell
$env:IR_PROCESSED_DIR = "data/processed"
pytest -q
```

## Core Workflow

1. Load documents with `DatasetLoader`.
2. Preprocess each document with `TextPreprocessor`.
3. Build the inverted index with `InvertedIndex`.
4. Preprocess user queries with `QueryProcessor`.
5. Use the shared token vocabulary for retrieval experiments.

## Notes

- The repository currently contains generated data under `data/`. If you want a clean rebuild, remove those outputs and rerun `scripts/process_docs.py`.
- `InvertedIndex` stores a basic term-to-document postings structure, document frequencies, document lengths, and total document count.

## License

No license file is currently included.

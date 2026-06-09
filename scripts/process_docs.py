import argparse
import os
import pickle
import sys
import time

import pandas as pd

sys.path.append('.')

from services.config import Config
from services.preprocessing.loader import DatasetLoader
from services.preprocessing.preprocessor import TextPreprocessor


def parse_args():
    parser = argparse.ArgumentParser(description="Load, save raw docs, and preprocess an IR dataset.")
    parser.add_argument(
        "--dataset",
        default=Config.DATASET_NAME,
        help="ir_datasets dataset name to load",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=Config.MAX_DOCS,
        help="Maximum number of documents to load",
    )
    parser.add_argument(
        "--raw-dir",
        default=Config.DATA_PATH,
        help="Directory to write raw dataset files",
    )
    parser.add_argument(
        "--processed-dir",
        default=Config.PROCESSED_PATH,
        help="Directory to write processed dataset files",
    )
    return parser.parse_args()


def log_step(message):
    print(f"\n=== {message} ===")


def save_processed_outputs(processed, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    generic_pkl = os.path.join(output_dir, "processed_docs.pkl")
    legacy_pkl = os.path.join(output_dir, "processed_docs_10k.pkl")
    generic_csv = os.path.join(output_dir, "processed_docs.csv")
    legacy_csv = os.path.join(output_dir, "processed_docs_10k.csv")

    with open(generic_pkl, 'wb') as f:
        pickle.dump(processed, f)
    with open(legacy_pkl, 'wb') as f:
        pickle.dump(processed, f)

    proc_df = pd.DataFrame([{k: v for k, v in d.items() if k != 'tokens'} for d in processed])
    proc_df.to_csv(generic_csv, index=False)
    proc_df.to_csv(legacy_csv, index=False)

    return generic_pkl, legacy_pkl, generic_csv, legacy_csv


def verify_outputs(paths):
    print("Output files:")
    for path in paths:
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {path} ({size} bytes)")


def main():
    args = parse_args()
    started_at = time.perf_counter()

    log_step("Configuration")
    print(f"Dataset      : {args.dataset}")
    print(f"Max docs     : {args.max_docs}")
    print(f"Raw dir      : {args.raw_dir}")
    print(f"Processed dir: {args.processed_dir}")

    log_step("Loading raw documents")
    loader = DatasetLoader(dataset_name=args.dataset, max_docs=args.max_docs)
    df = loader.load()
    print(f"Raw docs loaded: {len(df)}")

    log_step("Saving raw documents")
    raw_csv_path = os.path.join(args.raw_dir, "raw_docs.csv")
    loader.save_raw(df, raw_csv_path)

    log_step("Preprocessing documents")
    preprocessor = TextPreprocessor(use_stemming=False)
    processed = []
    total = len(df)
    for idx, row in df.iterrows():
        tokens = preprocessor.process(row['text'])
        processed.append({
            'doc_id': row['doc_id'],
            'original': row['text'],
            'tokens': tokens,
            'processed_text': ' '.join(tokens)
        })

        if (idx + 1) % 1000 == 0 or (idx + 1) == total:
            print(f"Processed {idx + 1}/{total} documents")

    log_step("Saving processed documents")
    outputs = save_processed_outputs(processed, args.processed_dir)
    verify_outputs([raw_csv_path, *outputs])

    elapsed = time.perf_counter() - started_at
    print(f"\nCompleted successfully in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()

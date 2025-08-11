from __future__ import annotations
import argparse
from typing import List
import pandas as pd
import pyarrow.parquet as pq
from transforms.base import BaseTransform


class KeepLatest200Features(BaseTransform):
    def __init__(self, input_path: str, output_path: str, keep_n: int = 200):
        super().__init__(input_path, output_path)
        self.keep_n = keep_n
        # Compute a stable column ordering from the Parquet schema once
        pf = pq.ParquetFile(self.input_path)
        schema = pf.schema_arrow
        all_cols: List[str] = [f.name for f in schema]
        essential = [c for c in all_cols if c.startswith("target") or c.startswith("era") or c.startswith("id")]
        features = [c for c in all_cols if c not in essential]
        self.final_columns: List[str] = essential + features[-self.keep_n:]

    def transform(self, df_chunk: pd.DataFrame) -> pd.DataFrame:
        # Reindex to the precomputed stable set; drop missing gracefully
        present = [c for c in self.final_columns if c in df_chunk.columns]
        return df_chunk[present]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", required=True)
    parser.add_argument("--output-data", required=True)
    parser.add_argument("--keep-n", type=int, default=200)
    args = parser.parse_args()

    tx = KeepLatest200Features(args.input_data, args.output_data, keep_n=args.keep_n)
    tx.run()


if __name__ == "__main__":
    main()

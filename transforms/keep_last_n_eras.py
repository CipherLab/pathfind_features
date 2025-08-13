from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

from transforms.base import BaseTransform

class KeepLastNEras(BaseTransform):
    """
    Stream a large Parquet and keep only rows whose era value is among the last N unique eras.
    Assumes an era-like column is present (default 'era') and that eras appear grouped or at least
    consistently per row.
    """
    def __init__(self, input_path: str, output_path: str, last_n: int, era_col: str = "era"):
        super().__init__(input_path, output_path)
        self.last_n = int(last_n)
        self.era_col = era_col
        if self.last_n <= 0:
            raise ValueError("last_n must be > 0")

        # Prepass to collect last N unique eras
        pf = pq.ParquetFile(self.input_path)
        from collections import deque
        dq = deque(maxlen=self.last_n)
        prev_last = None
        for batch in pf.iter_batches(batch_size=200_000, columns=[self.era_col]):
            ser = batch.to_pandas()[self.era_col]
            starts = ser != ser.shift(fill_value=prev_last)
            transitions = ser[starts]
            for v in transitions:
                dq.append(v)
            if not ser.empty:
                prev_last = ser.iloc[-1]
        self.selected_eras = set(dq)

    def transform(self, df_chunk: pd.DataFrame) -> pd.DataFrame:
        if self.era_col not in df_chunk.columns:
            return df_chunk.iloc[0:0]  # no era column; drop all to be safe
        return df_chunk[df_chunk[self.era_col].isin(self.selected_eras)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", required=True)
    parser.add_argument("--output-data", required=True)
    parser.add_argument("--last-n", type=int, required=True)
    parser.add_argument("--era-col", type=str, default="era")
    args = parser.parse_args()

    tx = KeepLastNEras(args.input_data, args.output_data, last_n=args.last_n, era_col=args.era_col)
    tx.run()


if __name__ == "__main__":
    main()

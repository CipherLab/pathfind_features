from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

class BaseTransform(ABC):
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    @abstractmethod
    def transform(self, df_chunk: pd.DataFrame) -> pd.DataFrame:
        """Implement the core transformation logic on a single chunk of data."""
        pass

    def run(self, batch_size=100_000):
        """Handles reading, chunking, transforming, and writing the data.

        Fixes:
        - Guard against input and output being the same file.
        - Stream Parquet output using ParquetWriter instead of re-reading and rewriting.
        """
        in_path = Path(self.input_path)
        out_path = Path(self.output_path)

        # Guard: prevent writing to the same file we're reading from
        try:
            if in_path.resolve() == out_path.resolve():
                raise ValueError("Output path must be different from input path to avoid corruption and infinite loops.")
        except Exception:
            # If resolve fails (e.g., non-existent out path yet), compare as-is
            if str(in_path) == str(out_path):
                raise ValueError("Output path must be different from input path to avoid corruption and infinite loops.")

        # Ensure output directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)

        parquet_file = pq.ParquetFile(str(in_path))

        # CSV streaming is simple via pandas
        if self.output_path.lower().endswith('.csv'):
            is_first_chunk = True
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                df_chunk = batch.to_pandas()
                transformed_chunk = self.transform(df_chunk)
                mode = 'w' if is_first_chunk else 'a'
                transformed_chunk.to_csv(str(out_path), index=False, mode=mode, header=is_first_chunk)
                is_first_chunk = False
            return

        # Default: Parquet streaming via ParquetWriter (efficient and stable size growth)
        writer: pq.ParquetWriter | None = None
        try:
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                df_chunk = batch.to_pandas()
                transformed_chunk = self.transform(df_chunk)
                # Convert to Arrow Table once per chunk
                table = pa.Table.from_pandas(transformed_chunk, preserve_index=False)
                if writer is None:
                    # Overwrite any existing file by recreating writer
                    # Some filesystems require explicit unlink to avoid partial appends
                    try:
                        if out_path.exists():
                            out_path.unlink()
                    except Exception:
                        # Best effort; ParquetWriter will still overwrite in most cases
                        pass
                    # Use compression and dictionary encoding to keep sizes efficient
                    writer = pq.ParquetWriter(
                        str(out_path),
                        table.schema,
                        compression="snappy",
                        use_dictionary=True,
                        write_statistics=True,
                    )
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()

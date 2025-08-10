from abc import ABC, abstractmethod
import pandas as pd
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
        """Handles reading, chunking, transforming, and writing the data."""
        parquet_file = pq.ParquetFile(self.input_path)
        is_first_chunk = True

        for batch in parquet_file.iter_batches(batch_size=batch_size):
            df_chunk = batch.to_pandas()
            transformed_chunk = self.transform(df_chunk)

            if self.output_path.endswith('.csv'):
                # Write to CSV in chunks
                if is_first_chunk:
                    transformed_chunk.to_csv(self.output_path, index=False, mode='w')
                    is_first_chunk = False
                else:
                    transformed_chunk.to_csv(self.output_path, index=False, mode='a', header=False)
            else:
                # For Parquet output, we would need to handle this differently,
                # potentially by writing to a temporary directory and then merging.
                # For now, we'll focus on the CSV case.
                if is_first_chunk:
                    transformed_chunk.to_parquet(self.output_path, index=False)
                    is_first_chunk = False
                else:
                    # This is inefficient for Parquet, but will work for now.
                    existing_df = pd.read_parquet(self.output_path)
                    combined_df = pd.concat([existing_df, transformed_chunk])
                    combined_df.to_parquet(self.output_path, index=False)

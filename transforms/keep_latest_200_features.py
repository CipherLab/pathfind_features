from transforms.base import BaseTransform
import pandas as pd

class KeepLatest200Features(BaseTransform):
    def transform(self, df_chunk: pd.DataFrame) -> pd.DataFrame:
        # Columns to always keep
        essential_cols = [col for col in df_chunk.columns if col.startswith('target') or col.startswith('era') or col.startswith('id')]
        
        # Get all feature columns (assuming they are not essential columns)
        feature_columns = [col for col in df_chunk.columns if col not in essential_cols]
        
        # Get the last 200 feature columns
        features_to_keep = feature_columns[-200:]
        
        # Combine essential columns with the selected features
        final_columns = essential_cols + features_to_keep
        
        return df_chunk[final_columns]

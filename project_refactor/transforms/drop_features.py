from transforms.base import BaseTransform
import pandas as pd

class DropFeatures(BaseTransform):
    def transform(self, df_chunk: pd.DataFrame) -> pd.DataFrame:
        features_to_drop = ['feature_1', 'feature_2']
        return df_chunk.drop(columns=features_to_drop, errors='ignore')
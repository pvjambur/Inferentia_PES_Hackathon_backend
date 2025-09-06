import pandas as pd
from typing import Dict, Any, Optional
import os
from utils.logging import get_logger

logger = get_logger(__name__)

class DataProcessor:
    """
    A class to handle all data processing and preprocessing tasks.
    """
    def __init__(self):
        logger.info("DataProcessor initialized.")

    def process_dataset(self, file_path: str) -> Dict[str, Any]:
        """
        Processes a dataset file to extract metadata such as schema and sample count.
        """
        logger.info(f"Processing dataset file: {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        df = None
        
        try:
            if file_extension == '.csv':
                df = pd.read_csv(file_path, nrows=5)
            elif file_extension == '.json':
                df = pd.read_json(file_path, orient='records', lines=True, nrows=5)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, nrows=5)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
        except Exception as e:
            logger.error(f"Error reading dataset file {file_path}: {e}")
            raise ValueError(f"Failed to read dataset file: {e}")
            
        try:
            num_samples = len(pd.read_csv(file_path)) if file_extension == '.csv' else len(pd.read_json(file_path, lines=True))
        except Exception as e:
            logger.warning(f"Could not determine the exact number of samples. Setting to None. Error: {e}")
            num_samples = None
        
        schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        logger.info("Dataset processing complete.")
        return {
            "num_samples": num_samples,
            "schema": schema
        }

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs a simple data cleaning and preprocessing step on a pandas DataFrame.
        """
        logger.info("Starting data preprocessing...")
        
        df_cleaned = df.dropna(thresh=len(df) * 0.7, axis=1)
        df_filled = df_cleaned.fillna(df_cleaned.mean(numeric_only=True))
        
        for col in df_filled.columns:
            if df_filled[col].dtype == 'object':
                df_filled[col] = pd.factorize(df_filled[col])[0]

        logger.info("Data preprocessing finished.")
        return df_filled
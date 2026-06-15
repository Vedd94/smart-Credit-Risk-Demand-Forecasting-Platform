import pandas as pd
import os
from pathlib import Path
from components.logger import logging
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        # df.fillna('', inplace=True)
        logging.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def basic_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        df = df.dropna().reset_index(drop = True)
        df.drop(columns='Unnamed: 0', inplace = True)
        logging.debug('Basic data preprocessing completed')
        return df
    except KeyError as e:
        logging.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error during preprocessing: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Advance preprocessing"""
    try:
        num_cols = ["Age", "Credit amount", "Duration"]
        nominal_cols = ["Sex", "Housing", "Purpose"]
        ordinal_cols = ["Saving accounts", "Checking account"]

        ohe = OneHotEncoder(sparse_output=False)

        encoded = ohe.fit_transform(df[nominal_cols])

        encoded_df = pd.DataFrame(
            encoded,
            columns=ohe.get_feature_names_out(nominal_cols),
            index=df.index
        )

        df = pd.concat(
            [df.drop(columns=nominal_cols), encoded_df],
            axis=1
        )
            
        orc = OrdinalEncoder()
        df[ordinal_cols] = orc.fit_transform(df[ordinal_cols])

        std = StandardScaler()
        df[num_cols] = std.fit_transform(df[num_cols])

        le_target = LabelEncoder()
        df["Risk"] = le_target.fit_transform(df["Risk"])

        logging.debug('Data preprocessing completed')
        return df
    except KeyError as e:
        logging.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(df: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        preprocessed_data_path = os.path.join(data_path, 'preprocessed')
        os.makedirs(preprocessed_data_path, exist_ok=True)
        df.to_csv(os.path.join(preprocessed_data_path, "preprocessed_train.csv"), index=False)
        logging.debug('Preprocessed data saved to %s', preprocessed_data_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
       
        data_path = Path("D:/GenAI/Barclays/smart-Credit-Risk-Demand-Forecasting-Platform/data/raw/train.csv")
        df = load_data(data_path)
        df = basic_preprocessing(df)
        final_df = preprocess_data(df)
        save_data(final_df, data_path='D:\GenAI\Barclays\smart-Credit-Risk-Demand-Forecasting-Platform\data')
        logging.info(f"Dataset Shape: {final_df.shape}")
        print(final_df.head())
        
    except Exception as e:
        logging.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
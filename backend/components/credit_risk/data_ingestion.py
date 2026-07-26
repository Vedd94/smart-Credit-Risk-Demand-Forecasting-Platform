import pandas as pd
from pathlib import Path
from components.logger import logging
from sklearn.model_selection import train_test_split
import os


def load_data(data_path):
    """
    Load data from a CSV file.
    """
    try:
        df = pd.read_csv(data_path)
        logging.debug(f"Data loaded from {data_path}")
        return df

    except pd.errors.ParserError as e:
        logging.error(f"Failed to parse the CSV file: {e}")
        raise

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error occurred while loading data: {e}")
        raise



def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logging.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        test_size = 0.2
        data_path = Path("D:/GenAI/Barclays/smart-Credit-Risk-Demand-Forecasting-Platform/notebooks/german_credit_data_100k.csv")

        df = load_data(data_path)
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=2)
        save_data(train_data, test_data, data_path='./data')
        logging.info(f"Dataset Shape: {df.shape}")
        print(df.head())

    except Exception as e:
        logging.error(
            f"Failed to complete the data ingestion process: {e}"
        )
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
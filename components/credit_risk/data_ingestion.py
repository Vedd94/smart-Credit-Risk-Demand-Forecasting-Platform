import pandas as pd
from pathlib import Path
from components.logger import logging


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


def main():
    try:
        data_path = Path("D:/GenAI/Barclays/smart-Credit-Risk-Demand-Forecasting-Platform/notebooks/german_credit_data_100k.csv")

        df = load_data(data_path)

        logging.info(f"Dataset Shape: {df.shape}")
        print(df.head())

    except Exception as e:
        logging.error(
            f"Failed to complete the data ingestion process: {e}"
        )
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
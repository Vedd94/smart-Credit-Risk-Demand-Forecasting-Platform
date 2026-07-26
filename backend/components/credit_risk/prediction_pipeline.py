import pandas as pd
from components.logger import logging
import joblib
import pickle
from pathlib import Path

# Project root = backend/
ROOT_DIR = Path(__file__).resolve().parents[2]

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ROOT_DIR / "models"


def main(user_input: pd.DataFrame):
    try:
        num_cols = ["Age", "Credit amount", "Duration"]
        nominal_cols = ["Sex", "Housing", "Purpose"]
        ordinal_cols = ["Saving accounts", "Checking account"]

        ohe = joblib.load(ARTIFACTS_DIR / "ohe.pkl")
        ohe_encoded = ohe.transform(user_input[nominal_cols])

        encoded_df1 = pd.DataFrame(
            ohe_encoded,
            columns=ohe.get_feature_names_out(nominal_cols),
            index=user_input.index
        )

        user_input = pd.concat(
            [user_input.drop(columns=nominal_cols), encoded_df1],
            axis=1
        )

        orc = joblib.load(ARTIFACTS_DIR / "ordinal_encoder.pkl")
        user_input[ordinal_cols] = orc.transform(user_input[ordinal_cols])

        scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
        user_input[num_cols] = scaler.transform(user_input[num_cols])

        with open(MODELS_DIR / "model.pkl", "rb") as m:
            model = pickle.load(m)

        prediction = model.predict(user_input.values)

        return "Bad" if prediction[0] == 0 else "Good"

    except KeyError as e:
        logging.error(f"Missing column in dataframe: {e}")
        raise

    except Exception:
        logging.exception("Prediction failed")
        raise
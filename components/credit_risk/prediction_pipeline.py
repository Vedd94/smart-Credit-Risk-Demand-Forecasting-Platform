import pandas as pd
from components.logger import logging
import joblib
import pickle

def main(user_input: pd.DataFrame):
    pass
    try:
        num_cols = ["Age", "Credit amount", "Duration"]
        nominal_cols = ["Sex", "Housing", "Purpose"]
        ordinal_cols = ["Saving accounts", "Checking account"]

        # user_input = {'Age':23,'Sex':'female','Job':2,'Housing':'own','Saving accounts':'quite rich','Checking account':'little','Credit amount':3543,'Duration':24,'Purpose':'radio/TV'}
        # user_input = pd.DataFrame([user_input])
        ohe = joblib.load("D:/GenAI/Barclays/smart-Credit-Risk-Demand-Forecasting-Platform/artifacts/ohe.pkl")
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

        orc = joblib.load("D:/GenAI/Barclays/smart-Credit-Risk-Demand-Forecasting-Platform/artifacts/ordinal_encoder.pkl")
        user_input[ordinal_cols] = orc.transform(user_input[ordinal_cols])

        scaler = joblib.load("D:/GenAI/Barclays/smart-Credit-Risk-Demand-Forecasting-Platform/artifacts/scaler.pkl")
        user_input[num_cols] = scaler.transform(user_input[num_cols])

        with open('models/model.pkl', 'rb') as m:
            model = pickle.load(m)
        
        prediction = model.predict(user_input.values)
        # print(type(prediction), prediction)
        if prediction == 0: return 'Bad'
        else:  return 'Good'

    except KeyError as e:
        logging.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logging.error('Prediction failed', e)
        raise


if __name__ == '__main__':
    main()
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from components.logger import logging
import os
from pathlib import Path
import pickle
import mlflow
import yaml


def load_params(params_path) -> dict:
    """ Load parameters from YAML file """

    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise



def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    :param file_path: Path to the CSV file
    :return: Loaded DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s with shape %s', file_path, df.shape)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except FileNotFoundError as e:
        logging.error('File not found: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_ml_model(X_train: np.ndarray, y_train: np.ndarray, params: dict):
    """
    Train the Decision tree, RandomForest model and XGBoost
    
    :param X_train: Training features
    :param y_train: Training labels

    :return: Trained models
    """
    model_dict = {}    
    try:
            if X_train.shape[0] != y_train.shape[0]:
                raise ValueError("The number of samples in X_train and y_train must be the same.")
            
            logging.info('Initializing Decision Tree model')
            dt = DecisionTreeClassifier(random_state=1, class_weight='balanced', min_samples_split=2, min_samples_leaf=2)
            grid_search_dt = GridSearchCV(estimator=dt, param_grid=params['model_training_dt'], cv=5, n_jobs=-1, verbose=2, scoring='roc_auc')
            grid_search_dt.fit(X_train, y_train)

            model_dict['grid_search_dt'] = grid_search_dt.cv_results_
            
            best_dt = grid_search_dt.best_estimator_
            logging.info('Decison tree trained. Best Params', grid_search_dt.best_params_)

            logging.info('Initializing Random Forest model')
            rfc = RandomForestClassifier(random_state=1, class_weight='balanced', n_jobs=-1,
                                        n_estimators=100,
                                        max_depth = 5,
                                        min_samples_split = 10,
                                        min_samples_leaf = 2)
            
            grid_search_rfc = GridSearchCV(estimator=rfc, param_grid=params['model_training_rfc'], cv=5, n_jobs=-1, verbose=2, scoring='roc_auc')
            grid_search_rfc.fit(X_train, y_train)
            
            model_dict['grid_search_rfc'] = grid_search_rfc.cv_results_
            best_rfc = grid_search_rfc.best_estimator_

            logging.info('Random Forest Classifier trained. Best params', grid_search_rfc.best_params_)

            logging.info('Initializing XGboost model')
            xgb = XGBClassifier(
                random_state=1,
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                eval_metric='logloss'
            )
            grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=params['model_training_rfc'], cv=5, n_jobs=-1, verbose=2, scoring='roc_auc')
            grid_search_xgb.fit(X_train, y_train)

            model_dict['grid_search_xgb'] = grid_search_xgb.cv_results_

            best_xgb = grid_search_xgb.best_estimator_
            logging.info('XGBoost model trained. Best Params:', grid_search_xgb.best_params_)

            voting_clf = VotingClassifier(
            estimators=[
                ('dt', best_dt),
                ('rfc', best_rfc),
                ('xgb', best_xgb)
            ],
            voting='soft'
            )

            voting_clf.fit(X_train, y_train)
            
            logging.info('ML models trained')
            
            return voting_clf, model_dict
    except ValueError as e:
            logging.error('ValueError during model training: %s', e)
            raise
    except Exception as e:
            logging.error('Error during model training: %s', e)
            raise

def save_model(model, file_path: str) -> None:
    """
    Save the trained model to a file.
    
    :param model: Trained model object
    :param file_path: Path to save the model file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)
    except FileNotFoundError as e:
        logging.error('File path not found: %s', e)
        raise
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        # params = {'n_estimators': 25, 'random_state': 2}
        params = load_params('D:/GenAI/Barclays/smart-Credit-Risk-Demand-Forecasting-Platform/components/credit_risk/params.yaml')
        preprocessed_data_path = Path("D:/GenAI/Barclays/smart-Credit-Risk-Demand-Forecasting-Platform/data/preprocessed/preprocessed_train.csv") 
        train_data = load_data(preprocessed_data_path)
        X_train = train_data.drop('Risk', axis = 1).values
        y_train = train_data['Risk'].values

        voting, model_dict = train_ml_model(X_train, y_train, params)
        
        model_save_path = Path("D:/GenAI/Barclays/Smart-Credit-Risk-Demand-Forecasting-Platform/models/model.pkl") 
        save_model(voting, model_save_path)

        return model_dict

    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()

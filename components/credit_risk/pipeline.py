from components.credit_risk import data_ingestion
from components.credit_risk import data_preprocessing
from components.credit_risk import model_training_2
from components.credit_risk import model_evaluation
import mlflow
import pickle

# Adding Training and mlflow pipeline
def main():
    data_ingestion.main()
    data_preprocessing.main()

    params_dict = model_training_2.main()
    metrices = model_evaluation.main()
    mlflow.set_experiment('Prediction pipeline')

    with mlflow.start_run() as parent:

        for GSCV_model in params_dict:
            for i in range(len(params_dict[GSCV_model]['params'])):
                with mlflow.start_run(nested=True) as child:
                    mlflow.log_params(params_dict[GSCV_model]['params'][i])
                    mlflow.log_metric("roc_auc", params_dict[GSCV_model]["mean_test_score"][i])

        mlflow.log_metric('accuracy', metrices['accuracy'])
        mlflow.log_metric('precision', metrices['precision'])
        mlflow.log_metric('recall', metrices['recall'])
        mlflow.log_metric('auc', metrices['auc'])

        with open("D:/GenAI/Barclays/smart-Credit-Risk-Demand-Forecasting-Platform/models/model.pkl", 'rb') as file:
            model = pickle.load(file)
        mlflow.sklearn.log_model(model, "Voting classifier")

if __name__ == '__main__':
    main()
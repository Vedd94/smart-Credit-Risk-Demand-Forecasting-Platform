from components.credit_risk import prediction_pipeline
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import Literal, Annotated
import pickle
import pandas as pd

# with open('models/model.pkl', 'rb') as f:
    # model = pickle.load(f)
class UserInput(BaseModel):

    age: Annotated[int, Field(..., gt=0, lt=120, description='Age of the user')]
    sex: Annotated[Literal['male', 'female'], Field(..., description='Gender of the user')]
    job: Annotated[int, Field(..., description='Number of jobs users did')]
    housing: Annotated[Literal['own', 'rent', 'free'], Field(..., description='Housing type of user')]
    saving_accounts: Annotated[Literal['little', 'moderate', 'rich', 'quite rich'], Field(..., description='Saving acc type')]
    checking_account: Annotated[Literal['little', 'moderate', 'rich'], Field(..., description='Checking acc type')]
    credit_amount: Annotated[int, Field(..., description= "input user's credit amount")]
    duration: Annotated[int, Field(..., description= "user credit duration")]
    purpose: Annotated[Literal['radio/TV', 'education', 'vacation/others', 'car','furniture/equipment', 'domestic appliances', 'business','repairs'], Field(..., description='Saving acc type')]
    
app = FastAPI()

@app.post('/predict')
def predict_risk(data: UserInput):

    input_df = pd.DataFrame([{
        'Age':data.age,
        'Sex':data.sex,
        'Job':data.job,
        'Housing':data.housing,
        'Saving accounts':data.saving_accounts,
        'Checking account':data.checking_account,
        'Credit amount':data.credit_amount,
        'Duration':data.duration,
        'Purpose':data.purpose
    }])

    prediction = prediction_pipeline.main(input_df)

    return JSONResponse(status_code=200, content={'predicted_category': prediction})

@app.get("/health")
def health():
    return {"status": "healthy"}
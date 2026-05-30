import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pydantic import BaseModel
from features import add_holidays, whole_data_processing

loaded_model = joblib.load(os.path.join(BASE_DIR, 'models/StoreSalesModel.pkl'))
xgb = loaded_model['xgb']
mlp = loaded_model['mlp']

X_time = joblib.load(os.path.join(BASE_DIR, 'models/X_timeDP.pkl'))
target_mean = joblib.load(os.path.join(BASE_DIR, 'models/TargetEncoding.pkl'))

class SalesRequest(BaseModel):
    date: str
    country: str
    store: str
    product: str

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_credentials = True,
    allow_methods = ['*'],
    allow_headers = ['*'],
)

@app.post('/predict-sales/')
async def predict_sales(request: SalesRequest):
    input_data = {'date': [request.date],
                  'country': [request.country],
                  'store': [request.store],
                  'product': [request.product]}
    
    df = pd.DataFrame(input_data)
    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.day_name()
    
    day_mapping = {'Friday': 0, 'Monday': 1, 'Saturday': 2, 'Sunday': 3,
                   'Thursday': 4, 'Tuesday': 5, 'Wednesday': 6}
    
    df['dayofweek_encoded'] = df['dayofweek'].map(day_mapping)

    df = whole_data_processing(df, X_time)
    df = df.merge(target_mean, on=['country', 'store', 'product'], how='left')
    drop_cols = ['id', 'date', 'country', 'store', 'product', 'dayofweek']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    expected_features = xgb.feature_names_in_
    X = X[expected_features]
    
    xgb_pred = xgb.predict(X)
    mlp_pred = mlp.predict(X)
    result = 0.8*xgb_pred + 0.2*mlp_pred
    final_result = float(result)
    return {'date': request.date, 'predicted_num_sold': max(float(0.0), final_result)}


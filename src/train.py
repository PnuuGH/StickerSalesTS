import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
import joblib
from features import data_processing, whole_data_processing, get_whole_time_feature


def training(features, target):
    X_train, X_val, y_train, y_val = train_test_split(features, target, shuffle = False)
    model_1 = XGBRegressor(n_estimators=1100, learning_rate=0.08, random_state=0)
    model_2 = make_pipeline(StandardScaler(), MLPRegressor(learning_rate_init = 0.001, hidden_layer_sizes = (32, 16, 8), alpha = 0.01, activation = 'tanh' , 
                                                                       max_iter = 1000, early_stopping= True, random_state = 0))
    model_1.fit(X_train, y_train)
    model_2.fit(X_train, y_train)
    xgb_pred = model_1.predict(X_val)
    mlp_pred = model_2.predict(X_val)
    
    result = (0.85 * xgb_pred) + (0.15* mlp_pred)
    MAPE = mean_absolute_percentage_error(y_val, result)
    return MAPE


def parameter_tuning(df, key):
    X, y = data_processing(df)
    TSCV = TimeSeriesSplit(n_splits = 3, test_size=365)
    if key == 'xgb':
        test_model = XGBRegressor(n_jobs = -1, eval_metric = 'rmse', 
                            random_state = 0, enable_categorical = False)
        param_dist = {
            'n_estimators': range(100, 1501, 100),
            'learning_rate': [0.01, 0.025, 0.05, 0.1, 0.2],
            'max_depth': [4, 5, 6, 7],
            'min_child_weight': [3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.3],
            'subsample': [0.6, 0.7, 0.8],
            'colsample_bytree': [0.6, 0.7, 0.8]
        }

    elif key == 'mlp':
        test_model = make_pipeline(StandardScaler(), MLPRegressor(max_iter= 1000, early_stopping= True, random_state = 0))

        param_dist = {
            'mlpregressor__hidden_layer_sizes': [(50,), (100, 50), (32, 16, 8)],
            'mlpregressor__activation': ['relu', 'tanh'],
            'mlpregressor__alpha': [0.0001, 0.01, 0.1],
            'mlpregressor__learning_rate_init': [0.001, 0.01]
        }
    
    else:
        print('Not in model list')

    pipeline_search = RandomizedSearchCV(estimator = test_model, param_distributions = param_dist, 
                                    n_iter = 30, scoring = 'neg_mean_absolute_percentage_error', cv = TSCV,
                                    verbose = 2, random_state = 0, n_jobs = -1)
    pipeline_search.fit(X, y)
    print('Best Parameters: ', pipeline_search.best_params_)
    print('Best Score: ', - pipeline_search.best_score_)

#parameter_tuning(df = df, key = 'mlp')

def main(df, df_test):
    X_time_global = get_whole_time_feature(df, df_test)
    joblib.dump(X_time_global, os.path.join(BASE_DIR, 'models/X_timeDP.pkl'))
    df_train = whole_data_processing(df, X_time_global)
    df_test = whole_data_processing(df_test, X_time_global)
    df['num_sold'] = df.groupby(['country', 'store', 'product'])['num_sold'].transform(lambda x: x.fillna(x.median()))
    
    target_mean = df_train.groupby(['country', 'store', 'product'])['num_sold'].mean().reset_index()
    target_mean = target_mean.rename(columns={'num_sold': 'target_mean_enc'})
    joblib.dump(target_mean, os.path.join(BASE_DIR, 'models/TargetEncoding.pkl'))

    df_train = df_train.merge(target_mean, on=['country', 'store', 'product'], how='left')
    df_test = df_test.merge(target_mean, on=['country', 'store', 'product'], how='left')
    
    df_train['num_sold'] = df_train['num_sold'].fillna(0)
    drop_cols = ['id', 'date', 'country', 'store', 'product', 'dayofweek', 'num_sold']

    X_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])
    y_train = df_train['num_sold']

    X_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])

    X_test = X_test[X_train.columns]
    model_1 = XGBRegressor(n_estimators=1100, learning_rate=0.08, random_state=0, early_stopping_rounds=0)
    model_2 = make_pipeline(SimpleImputer(strategy='constant', fill_value=0.0) , StandardScaler(), 
                            MLPRegressor(learning_rate_init = 0.001, hidden_layer_sizes = (32, 16, 8), alpha = 0.01, activation = 'tanh' , 
                                                                       max_iter = 1000, early_stopping= True, random_state = 0))
    
    model_1.fit(X_train, y_train)
    model_2.fit(X_train, y_train)
    xgb_pred = model_1.predict(X_test)
    mlp_pred = model_2.predict(X_test)

    test_predictions = (0.8 * xgb_pred) + (0.2 * mlp_pred)

    submission = pd.DataFrame({
        'id': df_test['id'],
        'num_sold': test_predictions
    })
    submission['num_sold'] = submission['num_sold'].clip(lower=0.0)
    submission.to_csv('submissions/submission_final.csv', index=False)
    print(X_train.columns)

    stacking_model = {'xgb': model_1, 'mlp': model_2}
    joblib.dump(stacking_model, os.path.join(BASE_DIR, 'models/StoreSalesModel.pkl'))

le = LabelEncoder()

if __name__ == '__main__':
    df = pd.read_csv('data/train.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['dayofweek'] = df['date'].dt.day_name()
    
    sale = df['num_sold']
    df = df.sort_values(by=['date', 'country', 'store', 'product']).reset_index(drop=True)
    df['dayofweek_encoded'] = le.fit_transform(df['dayofweek'])
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['is_weekend'] = df['dayofweek'].isin(['Saturday', 'Sunday']).astype(int)
    
    df_test = pd.read_csv('data/test.csv')
    df_test['date'] = pd.to_datetime(df_test['date'])
    df_test['dayofweek'] = df_test['date'].dt.day_name()
    df_test['dayofweek_encoded'] = le.transform(df_test['dayofweek'])
    
    main(df = df, df_test = df_test)
import pandas as pd
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
import holidays, joblib


def add_holidays(df):
    df = df.copy()
    df['is_holiday'] = 0
    country_list = df['country'].unique()
    for country in country_list:
        try:
            national_holidays = holidays.country_holidays(country)
            condition = (df['country'] == country) & (df['date'].isin(national_holidays))
            df.loc[condition, 'is_holiday'] = 1
        
        except NotImplementedError:
            print(f"Warning: No holiday data for {country}")
    return df


def get_time_feature(df):
    date_index = pd.period_range(df['date'].min(), df['date'].max(), freq='D')

    fourier = CalendarFourier(freq= 'A', order = 2)
    dp_index = date_index
    dp = DeterministicProcess(index = dp_index, order = 1, constant= True, 
                              seasonal= True, drop = True, period = 7, additional_terms=[fourier])
    
    X_time = dp.in_sample().reset_index().rename(columns={'index': 'date'})
    X_time['date'] = X_time['date'].dt.to_timestamp()
    return X_time

def data_processing(df):
    df = df.copy()
    df = add_holidays(df)
    df['year'] = df['date'].dt.year
    X_time = get_time_feature(df)
    df = df.merge(X_time, on='date', how='left')
    
    df['num_sold'] = df.groupby(['country', 'store', 'product'])['num_sold'].transform(lambda x: x.fillna(x.median()))
    df['target_mean'] = df.groupby(['country', 'store', 'product'])['num_sold'].transform('mean')
   
    df_combined = pd.concat([df[X_time.columns.drop('date')],
                      df[['dayofweek_encoded', 'num_sold', 'month', 'day_of_year', 
                           'is_holiday' , 'is_weekend', 'target_mean']]], axis=1)
    
    df_combined = df_combined.dropna().reset_index(drop=True)
    features = df_combined.drop(columns=['num_sold'])
    target = df_combined['num_sold']
    return features, target

#Below this line is for the unseen data set (test.csv) feature engineering, and above that is for LIB testing


def get_whole_time_feature(df, df_test):
    date_index = pd.period_range(df['date'].min(), df_test['date'].max(), freq='D')

    fourier = CalendarFourier(freq= 'A', order = 2)
    dp_index = date_index
    dp = DeterministicProcess(index = dp_index, order = 1, constant= True, 
                              seasonal= True, drop = True, period = 7, additional_terms=[fourier])
    X_whole_time = dp.in_sample().reset_index().rename(columns={'index': 'date'})

    X_whole_time['date'] = X_whole_time['date'].dt.to_timestamp()
    
    return X_whole_time

def whole_data_processing(df, X_time_df):
    df = df.copy()
    df = add_holidays(df)
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['is_weekend'] = df['date'].dt.day_name().isin(['Saturday', 'Sunday']).astype(int)
    df = df.merge(X_time_df, on='date', how='left')

    return df
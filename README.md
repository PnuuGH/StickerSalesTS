
"Download train.csv and test.csv from Kaggle 
Playground Series S5E1 and place in data/ folder"


1. Project title
    This project is forecasting sticker sales using Time Series Machine Learning using dataset from Kaggle competition.

2. Problem Statement
    The challenge is test set is 3 years ahead of train with different categories such as product, store and country. Also there are some categories having their entire data as missing value that we have to analyze and decide what shall we do with them.

3. Dataset
    - Source (Kaggle Playground S5E1)
    - Training: 2010-2016 (230131 rows)
    - Test: 2017-2019 (98551 rows)
    - 6 countries, 3 stores, 5 products

4. Approach & Key Decisions
    - I decided to not use Linear Regression nor Ridge model since the dataset contains different product category and each of them has their own trend and seasonality. As such, 1 linear model can not fit all trend and seasonality categories that is the reason why I decided to use gradient model like XGBoost having no problem to handle with several categories.

    - Another important decision is I decided not to use lag feature (previous sales in x days) because the timeframe for prediction is 3 years which is relatively long compared to the train dataset's frame (7 years). Since lag features require historical num_sold which doesn't exist in test set since test is 3 years ahead of train with no overlapping sales history. Instead, I used Fourier series as seasonality together with seasonal indicator. The periodogram shows dominant biweekly (26) seasonality. Since I already use seasonal indicators (period=7) to capture weekly patterns, I added annual Fourier terms (freq='A', order=2) to capture longer-term annual patterns that seasonal indicators alone cannot capture compensating for the absence of lag features.

5. Features Table 

    Feature                             Type                            Description
    
    const                               Time                            Intercept from DeterministicProcess
    trend                               Time                            Linear time progression
    s(2,7) to s(7,7)                    Time                            Seasonal indicator: Monday to Sunday (drop s(1,7) to avoid multicollinearity)
    sin(1,year), cos(1,year)            Time                            Annual Fourier term order 1 which captures yearly cycle
    sin(2,year), cos(2,year)            Time                            Annual Fourier term order 2 which captures sub-yearly cycle
    dayofweek_encoded                   Calendar                        encode day of week into number 0-6
    month                               Calendar                        determine a month
    day_of_year                         Calendar                        determine a day from 1-365
    is_holiday                          Calendar                        determine whether it is national holiday or not
    is_weekend                          Calendar                        determine whether it is weekend or not
    target_mean                         Encoding                        use the mean of particular store and product sold in certain country to be a feature
    
6. Model Architecture
    - I used ensemble model including XGBoost together with MultiLayer Perception (MLP neural network) because XGB is tree-based model while MLP is neural network making their learning and decision different. Consequently, the MLP will cover XGB weaknesses and vice versa.
    - The weight XGB : MLP = 0.85 : 0.15 is from experiment; I have tried MLP weight between 0.15-0.25 they all perform well;however, 0.15 weight makes least difference between public MAPE (Mean absolute percentage error) and private MAPE making this weight the most stable one.

7. Results
    - MAPE validation score = 0.0987
    - Kaggle leaderboard score: Public score = 0.13328
                                Private score = 0.15572

    The gap between validation and leaderboard scores reflects temporal distribution shift; the model was validated approcimately on 2015-2016 data but evaluated on 2017-2019. Structural features like trend and seasonality assume historical patterns repeat, which introduces error when real-world behavior drifts over a 3-year horizon. Heavy hyperparameter tuning increased this gap, suggesting the baseline parameters generalize better across time periods than tuned ones.
                        
8. EDA Visualizations
    - Actual Sales VS Predicted Sales(Baseline).png: This figure is the result from prototype to see how it performs on partial dataset
    - Actual VS Trend(Baseline).png: This figure compares the actual sales of partial dataset with the two weeks average to visualize the trend and decide the order of polynomial trend line (order = 1 in this case)
    - Periodogram.png: This figure shows the variance of sales in different time frame (visualize to decide the frequency and order of Fourier series).
    - Weekly_Seasonality.png: This boxplot tells whether the different days influence the total sales or not (used to select seasonality)

9. Project Structure
    
    StickerSaleTS/
    ├── src/
    │   ├── features.py
    │   ├── train.py
    │   └── predict.py
    ├── notebooks/
    │   └── prototype.ipynb
    ├── data/
    │   └── plot/
    ├── models/
    ├── submissions/
    └── requirements.txt
    └── README.md

10. Setup & Usage
    - pip install -r requirements.txt
    - Place train.csv and test.csv in data/ folder
    - python src/train.py
    - uvicorn src.predict:app --reload
    - Open http://localhost:8000/docs

11. What I learnt from this project
    - Using lag feature does not work for long timeframe prediction since not only we do not have historical data, but also were we to use recursive prediction or direct prediction for lag, it'd cause compound error that could mess the whole thing up.
    - Linear models like LinearRegression or Ridge cannot be used to predict the target with multiple categories like product, store or country since every type has their own trend and seasonality, so 1 single linear equation is not enough to fit all categories. As a result using just 1 linear model to predict high cardinality target would cause massive error.
    - Since this is the first time I put the machine learning in to real practice, I learnt the basic concept of API, and deployed fastAPI for the first time. Additionally, I've learnt how to organize the project into the clean structure because I would like to put it into my first portfolio's work.
    - sys.path must be configured before imports when splitting project into multiple files across subdirectories
    - Target encoding must be fit on train only then mapped to test set; computing on full dataset introduces data leakage from future into training.
    - DeterministicProcess requires unique date index since panel data with multiple groups cannot index DP directly, requires building on unique dates then merging back. Also, single DeterministicProcess works for panel data because DP generates calendar signals not sales signals. Group differences are in scale not seasonal pattern shape, so one DP shared across all groups is valid.

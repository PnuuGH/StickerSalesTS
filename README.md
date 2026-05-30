1. Project title
    This project is forecasting sticker sales using Time Series Machine Learning using dataset from Kaggle competition.

2. Problem Statement
    The challenge is test set is 3 years ahead of train with different categories such as product, store and country.

3. Dataset
    - Source (Kaggle Playground S5E1)
    - Size (rows, countries, stores, products)
    - Train/test date range

4. Approach & Key Decisions
    - I decided to not use Linear Regression nor Ridge model since the dataset contains different product category and each of them has their own trend and seasonality. As such, 1 linear model can not fit all trend and seasonality categories that is the reason why I decided to use gradient model like XGBoost having no problem to handle with several categories.

    - Another important decision is I decided not to use lag feature (previous sales in x days) because the timeframe for prediction is 3 years which is relatively long compared to the train dataset's frame (5 years). Since lag features require historical num_sold which doesn't exist in test set since test is 3 years ahead of train with no overlapping sales history. Instead, I used Fourier series as seasonality together with seasonal indicator. The periodogram shows dominant biweekly (26) seasonality. Since I already use seasonal indicators (period=7) to capture weekly patterns, I added annual Fourier terms (freq='A', order=2) to capture longer-term annual patterns that seasonal indicators alone cannot capture compensating for the absence of lag features.

5. Features Table 

    Feature                             Type                            Description
    
    const                               Time                            Intercept from DeterministicProcess
    trend                               Time                            Linear time progression
    s(2,7) to s(7,7)                    Time                            Seasonal indicator: Monday to Sunday (drop s(1,7) to avoid multicollinearity)
    sin(1,year), cos(1,year)            Time                            Fourier Series
    sin(2,year), cos(2,year)            Time                            Fourier Series
    dayofweek_encoded                   Calendar                        encode day of week into number 0-6
    month                               Calendar                        determine a month
    day_of_year                         Calendar                        determine a day from 1-365
    is_holiday                          Calendar                        determine whether it is national holiday or not
    is_weekend                          Calendar                        determine whether it is weekend or not
    target_mean                         Encoding                        use the mean of particular store and product sold in certain country to be a feature
    
6. Model Architecture
    - I used ensemble model including XGBoost together with MultiLayerPerception (MLP neural network) because XGB is tree-based model while MLP is neural network making their learning and decision different. Consequently, the MLP will cover XGB weaknesses and vice versa.
    - The weight XGB : MLP = 0.85 : 0.15 is from experiment; I have tried MLP weight between 0.15-0.25 they all perform well;however, 0.15 weight makes least difference between public MAPE (Mean absolute percentage error) and private MAPE making this weight the most stable one.

7. Results
    - MAPE validation score = 0.0987
    - Kaggle leaderboard score: Public score = 0.13328
                                Private score = 0.15572
                        
8. EDA Visualizations
    - Actual Sales VS Predicted Sales(Baseline).png: This figure is the result from prototype to see how it performs on partial dataset
    - Actual VS Trend(Baseline).png: This figure compares the actual sales of partial dataset with the two weeks average to visualize the trend and decide the order of polynomial trend line (order = 1 in this case)
    - Periodogram.png: This figure shows the varience of sales in different time frame (visualize to decide the frequency and order of Fourier series).
    - Weekly_Seasonality.png: This boxplot tells whether the different days influence the total sales or not (used to select seasonality)

9. Project Structure
    - folder tree

10. Setup & Usage
    - Run the features.py and train.py respectively to prepare the feature, train model and save model into pkl file
    - To open fastAPI, type "uvicorn src.predict:app --reload" in predict.py file's terminal to launch it 
    - Then, open the link acquired from the terminal in the browser and add /docs to get easily-visualized interface
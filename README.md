# Stocks
I want to predict the market with AI; just like everyone and their nan. Also, this is an old project so I aim to redo it with a bunch of bells and whistles (maybe a CNN working over ~500 stocks, like the S&P).

## Model
MLP with SDG.

## Aim
To predict monday morning stock price given data from the previous week.

## Data
Sourced from Python's yfinance.
For now only from apple, though it is generalisable.

Hourly Open, Close, Volume, High, low values, from monday to friday, except the last hour on friday to give time to buy the stock.
Stock price after the weekend; that is, Monday morning.

## Issues
1. I only have 65 training points because yfinance only provides hourly data for 2 years from today, so I have to use a lot of epochs to get decent accuracy, however, this will definitely cause overfitting.
2. All items in the validation set are increasing, this seems suspicious, data needs to be inspected.
3. Predictions are not accurate in predicting stock price.

## Moving Forward
1. Look up neptune.ai article about AI in stock prediction, mentions something about estimated moving average (EMA).
2. Find another python library or API with more access to data.
3. Use folds for training, validation, testing.

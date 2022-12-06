# Stocks

## Model
Deep learning neural network (DNN) with stochastic gradient descent.

## Aim
To predict monday morning stock price given data from the previous week.

## Data
Sourced from Python's yfinance.
For now only from apple, though it is generalisable.

Hourly Open, Close, Volume, High, low values, from monday to friday, except the last hour on friday to give time to buy the stock.
Stock price after the weekend; that is, Monday morning.

## Issues
1. I only have 65 training points because yfinance only provides hourly data for 2 years from today, so I have to use a lot of epochs to get decent accuracy, however, this may cause overfitting.
2. All items in the validation set are increasing, this seems suspicious, data needs to be inspected.
3. Predictions are not accurate in predicting stock price; they may become more accurate as a classifer for 2 labels (up, down), or 3 labels (up by 1+, |change| < 1, down by 1+).

## Moving Forward
1. Look up neptune.ai article about AI in stock prediction, mentions something about estimated moving average (EMA).
2. Find another python library with more access to data.
3. Use folds for training, validation, testing.
4. Convert DNN into a classification model.


### Github readme guide
https://medium.com/swlh/how-to-make-the-perfect-readme-md-on-github-92ed5771c061

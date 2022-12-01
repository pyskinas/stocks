tickers_list = ["aapl", "goog", "amzn", "BAC", "BA"]
tickers_data = {}

import numpy as np

# for ticker in tickers_list:
#     ticker_object = yf.Ticker(ticker)
    
#     # Convert Ticker() output from dict to df
#     temp = pd.DataFrame.from_dict(ticker_object.info, orient="index")
#     temp.reset_index(inplace=True)
#     temp.columns = ["Attribute", "Recent"]
    
#     # Add ticker to dictionary
#     tickers_data[ticker] = temp

# ticker = yf.Ticker(tickers_list[0])
# ticker_history = ticker.history( period="6d", interval="5m")
# print(ticker_history)

## Saving Files to pc
# x = [[1,2], [3,4]]
# np.savetxt("text.csv", x, delimiter=",")

M = np.arange(9).reshape((3,3))
v = np.random.randn(3,1)
v2 = np.random.randn(3,1)
print(M)
print(M[:2])
print(v)
print(M.T.dot(v) + v2)
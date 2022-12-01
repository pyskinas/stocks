from tempfile import TemporaryFile

# Stock Data
import yfinance as yf

# Dataftames, useful for numpy
import pandas as pd

# Matrices to represent stocks
import numpy as np

# To get today's date.
import datetime

def main():
    
    # Get the day of the week, 0 is monday, 6 is sunday
    day = datetime.datetime.now().weekday() # String
    
    # Use .strftime("%Y-%m-%d") to convert date to a string
    # Get the start date as the previous monday
    # Get the end date as the previous saturday
    startDate = datetime.date.today() - datetime.timedelta(days=(7 + day)) # date object
    endDate = datetime.date.today() - datetime.timedelta(days=(2 + day)) # date object
    
    # Initialise the ticker
    ticker_name = "aapl"
    ticker = yf.Ticker(ticker_name)
    
    # Number of data points
    N = 103
    
    # Dataset
    Data = []

    for i in range(N):
        print(i)
        # Get the start, monday, and end, Saturday, for every week up to N weeks ago
        startDateVar = startDate - datetime.timedelta(days=(i*7))
        endDateVar = endDate - datetime.timedelta(days=(i*7))
        
        weekData = WeeklyData(startDateVar, endDateVar, ticker)
        
        if weekData == 0:
            continue
        
        Data.append(weekData)
        pass
    pass
    
    # Save the data
    np.savetxt("Data.csv", Data, delimiter=",")


def WeeklyData(startDate, endDate, ticker):
    # 5 properties, 5 days a week, 7 hours a day, except the last hour of friday. 
    # Thats 5*5*7 -5 = 170, then the output, so 170 + 1 = 171
    weeklyData = []
    
    # Get the daily data, clean it up, and add it onto the output array
    for i in range(5):
        # Get the start date and end date for just 1 day
        startD = startDate + datetime.timedelta(days=i)
        endD = startDate + datetime.timedelta(days=1+i)
        
        # Convert days to string
        startSTR = startD.strftime("%Y-%m-%d")
        endSTR = endD.strftime("%Y-%m-%d")
         
        # print(f"Start Date: {startSTR}, End Date: {endSTR} ")
        df = pd.DataFrame(ticker.history(start=startSTR, end=endSTR, interval="1h"))
        
        # Get number of hours recorded in a day
        N = len(df)
        
        # If you get 0, or 4 or some shit, then just drop the data by continuing the for
        # loop into the next iteration
        if N not in [6,7,8]:
            continue
        
        # For every property I'm interested in. 
        keys = ["Open", "Close", "High", "Low", "Volume"]
        for key in keys:
            temp = np.array(df[key])
            n = len(temp)
            if n == 6:
                temp.append(temp[5])
            elif n == 8:
                temp = temp[:7]
            
            # If friday, remove last hour value
            if i == 4:
                temp = temp[:6]
            
            weeklyData.extend(temp)
    
    # All input data needs to be of the same sort
    if len(weeklyData) != 170:
        return 0
    
    # The output of the NN will be the open price of the stock on the following monday
    today = startDate + datetime.timedelta(days=7)
    today1 = startDate + datetime.timedelta(days=8)
    today = today.strftime("%Y-%m-%d")
    today1 = today1.strftime("%Y-%m-%d")
    
    # get the opening price for the following monday, if monday data is missing, kill data
    try:
        nextDayOpen = np.array(pd.DataFrame(ticker.history(start=today, end=today1, interval="1h"))["Open"])[0]
    except IndexError:
        return 0 
    
    # Add it on as the output, in the 0th index
    weeklyData.insert(0,nextDayOpen)
    
    return weeklyData

main()
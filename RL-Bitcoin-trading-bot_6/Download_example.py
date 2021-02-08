import bitfinex
import datetime
import time
import pandas as pd

# Create api instance of the v2 API
api_v2 = bitfinex.bitfinex_v2.api_v2()

# Define query parameters
pair = 'BTCUSD' # Currency pair of interest
TIMEFRAME = '1h'#,'4h','1h','15m','1m'

# Define the start date
t_start = datetime.datetime(2018, 9, 1, 0, 0)
t_start = time.mktime(t_start.timetuple()) * 1000

# Define the end date
t_stop = datetime.datetime(2020, 10, 1, 0, 0)
t_stop = time.mktime(t_stop.timetuple()) * 1000

# Download OHCL data from API
result = api_v2.candles(symbol=pair, interval=TIMEFRAME, limit=1000, start=t_start, end=t_stop)

# Convert list of data to pandas dataframe
names = ['Date', 'Open', 'Close', 'High', 'Low', 'Volume']
df = pd.DataFrame(result, columns=names)
df['Date'] = pd.to_datetime(df['Date'], unit='ms')

# we can plot our downloaded data
import matplotlib.pyplot as plt
plt.plot(df['Open'],'-')
plt.show()

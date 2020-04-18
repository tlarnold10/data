# 1I5ZNRGCLS9VRF96
import requests
import json
import pdb
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=MSFT&outputsize=full&apikey=1I5ZNRGCLS9VRF96'

response = requests.get(url)
json_data = response.json()

date_array = []
open_array = []
high_array = []
low_array = []
close_array = []
ad_close_array = []
volume_array = []
div_array = []
split_array = []

for day in json_data['Time Series (Daily)']:
    date_array.append(day)
    # print('Open: ' + json_data['Time Series (Daily)'][day]['1. open'])
    open_array.append(json_data['Time Series (Daily)'][day]['1. open'])
    # print('High: ' + json_data['Time Series (Daily)'][day]['2. high'])
    high_array.append(json_data['Time Series (Daily)'][day]['2. high'])
    # print('Low: ' + json_data['Time Series (Daily)'][day]['3. low'])
    low_array.append(json_data['Time Series (Daily)'][day]['3. low'])
    # print('Clos: ' + json_data['Time Series (Daily)'][day]['4. close'])
    close_array.append(json_data['Time Series (Daily)'][day]['4. close'])
    # print('Adjusted Close: ' + json_data['Time Series (Daily)'][day]['5. adjusted close'])
    ad_close_array.append(float(json_data['Time Series (Daily)'][day]['5. adjusted close']))
    # print('Volume: ' + json_data['Time Series (Daily)'][day]['6. volume'])
    volume_array.append(json_data['Time Series (Daily)'][day]['6. volume'])
    # print('Dividend Amount: ' + json_data['Time Series (Daily)'][day]['7. dividend amount'])
    div_array.append(json_data['Time Series (Daily)'][day]['7. dividend amount'])
    # print('Split Coefficient: ' + json_data['Time Series (Daily)'][day]['8. split coefficient'])
    split_array.append(json_data['Time Series (Daily)'][day]['8. split coefficient'])

data_dict = {}
data_dict['date'] = date_array 
data_dict['open'] = open_array
data_dict['high'] = high_array
data_dict['low'] = low_array
data_dict['close'] = close_array
data_dict['ad_close'] = ad_close_array
data_dict['volume'] = volume_array
data_dict['dividend'] = div_array
data_dict['split'] = split_array
# pdb.set_trace()
stock_df = pd.DataFrame(data_dict)

stock_df.plot(kind='line',x='date',y='ad_close')
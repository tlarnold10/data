# 1I5ZNRGCLS9VRF96
import requests
import json
import pdb
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser(description='Create a chart for a given stock ticker')
parser.add_argument('--ticker', help='The stock ticker you are wanting to look at.')

args = parser.parse_args()

beginning_url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=' 
end_url = '&outputsize=full&apikey=1I5ZNRGCLS9VRF96'

url = beginning_url + args.ticker + end_url

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
    open_array.append(float(json_data['Time Series (Daily)'][day]['1. open']))
    # print('High: ' + json_data['Time Series (Daily)'][day]['2. high'])
    high_array.append(json_data['Time Series (Daily)'][day]['2. high'])
    # print('Low: ' + json_data['Time Series (Daily)'][day]['3. low'])
    low_array.append(json_data['Time Series (Daily)'][day]['3. low'])
    # print('Clos: ' + json_data['Time Series (Daily)'][day]['4. close'])
    close_array.append(json_data['Time Series (Daily)'][day]['4. close'])
    # print('Adjusted Close: ' + json_data['Time Series (Daily)'][day]['5. adjusted close'])
    ad_close_array.append(float(json_data['Time Series (Daily)'][day]['5. adjusted close']))
    # print('Volume: ' + json_data['Time Series (Daily)'][day]['6. volume'])
    volume_array.append(float(json_data['Time Series (Daily)'][day]['6. volume']))
    # print('Dividend Amount: ' + json_data['Time Series (Daily)'][day]['7. dividend amount'])
    div_array.append(json_data['Time Series (Daily)'][day]['7. dividend amount'])
    # print('Split Coefficient: ' + json_data['Time Series (Daily)'][day]['8. split coefficient'])
    split_array.append(json_data['Time Series (Daily)'][day]['8. split coefficient'])

data_dict = {}
data_dict['date'] = reversed(date_array) 
data_dict['open'] = reversed(open_array)
data_dict['high'] = reversed(high_array)
data_dict['low'] = reversed(low_array)
data_dict['close'] = reversed(close_array)
data_dict['ad_close'] = reversed(ad_close_array)
data_dict['volume'] = reversed(volume_array)
data_dict['dividend'] = reversed(div_array)
data_dict['split'] = reversed(split_array)
stock_df = pd.DataFrame(data_dict)

stock_df.plot(kind='line',x='date',y='ad_close')

plt.show()

model = LinearRegression().fit(stock_df['open'].values.reshape((-1, 1)), stock_df['ad_close'])
print(f'Intercept: {model.intercept_}')
print(f'Coefficient: {model.coef_}')
correlation = stock_df['ad_close'].corr(stock_df['open'])
print(f'Correlation: {correlation}')
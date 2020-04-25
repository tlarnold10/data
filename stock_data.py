# 1I5ZNRGCLS9VRF96
import requests
import json
import pdb
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures 

parser = argparse.ArgumentParser(description='Create a chart for a given stock ticker')
parser.add_argument('--ticker', help='The stock ticker you are wanting to look at.')

args = parser.parse_args()

beginning_url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=' 
end_url = '&outputsize=full&apikey=1I5ZNRGCLS9VRF96'

url = beginning_url + args.ticker + end_url
try:
    response = requests.get(url)
except:
    print(f'{args.ticker} is not a valid stock ticker. Try again')
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
periods = []

for day in json_data['Time Series (Daily)']:
    date_array.append(day)
    # print('Open: ' + json_data['Time Series (Daily)'][day]['1. open'])
    open_array.append(float(json_data['Time Series (Daily)'][day]['1. open']))
    # print('High: ' + json_data['Time Series (Daily)'][day]['2. high'])
    high_array.append(float(json_data['Time Series (Daily)'][day]['2. high']))
    # print('Low: ' + json_data['Time Series (Daily)'][day]['3. low'])
    low_array.append(float(json_data['Time Series (Daily)'][day]['3. low']))
    # print('Clos: ' + json_data['Time Series (Daily)'][day]['4. close'])
    close_array.append(float(json_data['Time Series (Daily)'][day]['4. close']))
    # print('Adjusted Close: ' + json_data['Time Series (Daily)'][day]['5. adjusted close'])
    ad_close_array.append(float(json_data['Time Series (Daily)'][day]['5. adjusted close']))
    # print('Volume: ' + json_data['Time Series (Daily)'][day]['6. volume'])
    volume_array.append(float(json_data['Time Series (Daily)'][day]['6. volume']))
    # print('Dividend Amount: ' + json_data['Time Series (Daily)'][day]['7. dividend amount'])
    div_array.append(float(json_data['Time Series (Daily)'][day]['7. dividend amount']))
    # print('Split Coefficient: ' + json_data['Time Series (Daily)'][day]['8. split coefficient'])
    split_array.append(float(json_data['Time Series (Daily)'][day]['8. split coefficient']))

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

# Create a new column for the number of periods (days)
row_count = stock_df.shape[0]
for i in range(0, row_count):
    periods.append(i)

stock_df['period'] = periods
stock_df.plot(kind='line',x='date',y='ad_close')

print(f'\n##################################################################')
print(f'#             Let\'s get some descriptive statistics              #')
print(f'##################################################################')
print(f'Row Count: {row_count}')
column_count = stock_df.shape[1]
print(f'Column Count: {column_count}')
ad_close_description = stock_df['ad_close'].describe()
print(f'Adjusted Close Description:\n{ad_close_description}')  
plt.title('Adjusted Close for: ' + args.ticker)  
plt.xlabel('Date')
plt.ylabel('Adjused Closed')
plt.show()

# Linear Regression Things 
x = stock_df['period'].values.reshape((-1, 1))
y = stock_df['ad_close']
model = LinearRegression().fit(x, y)
print(f'\n############################################################')
print(f'# Get some of the info about the linear regression testing #')
print(f'############################################################')
print(f'Intercept: {model.intercept_}')
print(f'Coefficient: {model.coef_}')
correlation = stock_df['ad_close'].corr(stock_df['period'])
print(f'Correlation: {correlation}')

# Now let's do it the way we are supposed to do it, 
# create the model with 80% and use the other 20% for testing
x_train, x_test, y_train, y_test = train_test_split(stock_df['period'].values.reshape((-1, 1)), 
                                        stock_df['ad_close'], test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
prediction_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Get accuracy of the model. 
print(f'\n################################################################')
print(f'#   Just developed the model using a test section of the data  #')
print(f'################################################################')
print(f'Actual vs. Predicted:\n{prediction_df}')
print(f'Mean Absolute Error: {metrics.mean_absolute_error(y_test, y_pred)}')  
print(f'Mean Squared Error: {metrics.mean_squared_error(y_test, y_pred)}')  
print(f'Root Mean Squared Error: {np.sqrt(metrics.mean_squared_error(y_test, y_pred))}')

# plot stuff and things
plt.scatter(x_test, y_test,  color='gray')
plt.plot(x_test, y_pred, color='red', linewidth=2)
plt.show()

# Multiple linear regression 
# model = LinearRegression().fit(stock_df['open'].values.reshape((-1, 1)), stock_df['ad_close'])
print(f'\n############################################################')
print(f'#               Multiple linear regression                 #')
print(f'############################################################')


# Polynomial regression 
# model = LinearRegression().fit(stock_df['open'].values.reshape((-1, 1)), stock_df['ad_close'])
print(f'\n############################################################')
print(f'#                 Polynomial regression                    #')
print(f'############################################################')

x = stock_df['period'].values.reshape((-1,1))
y = stock_df['ad_close']
poly = PolynomialFeatures(degree = 4) 
X_poly = poly.fit_transform(x) 
  
poly.fit(X_poly, y) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y) 

plt.scatter(x, y, color = 'blue') 
  
plt.plot(x, lin2.predict(poly.fit_transform(x)), color = 'red') 
plt.title('Polynomial Regression') 
plt.xlabel('Temperature') 
plt.ylabel('Pressure') 
  
plt.show() 
# 1I5ZNRGCLS9VRF96
import requests
import json
import pdb

url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=MSF&outputsize=full&apikey=1I5ZNRGCLS9VRF96'

response = requests.get(url)
json_data = response.json()
for day in json_data['Time Series (Daily)']:
    print('Open: ' + json_data['Time Series (Daily)'][day]['1. open'])
    print('High: ' + json_data['Time Series (Daily)'][day]['2. high'])
    print('Low: ' + json_data['Time Series (Daily)'][day]['3. low'])
    print('Clos: ' + json_data['Time Series (Daily)'][day]['4. close'])
    print('Adjusted Close: ' + json_data['Time Series (Daily)'][day]['5. adjusted close'])
    print('Volume: ' + json_data['Time Series (Daily)'][day]['6. volume'])
    print('Dividend Amount: ' + json_data['Time Series (Daily)'][day]['7. dividend amount'])
    print('Split Coefficient: ' + json_data['Time Series (Daily)'][day]['8. split coefficient'])
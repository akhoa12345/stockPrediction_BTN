import time
import external_stock_data

while True:
    data = external_stock_data.getLatestStockPrice('BTC')
    print(data)
    time.sleep(3)
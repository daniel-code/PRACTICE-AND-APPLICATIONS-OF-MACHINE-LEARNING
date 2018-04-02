"""
利用股票數值預測未來股價
python 3.5
Author:daniel-code
Date:2018.04.02
"""
# coding=UTF-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# 參數設定
STOCK_FILE = 'Apple'
TEST_NUM = 100
# 讀取資料
data = pd.read_csv(STOCK_FILE + '.csv')
data_list = data.values

x = data_list[:-1, 1:4]
y = data_list[1:, 4]
print(data_list)
print(x)
print(y)
x_train, x_test = x[:-TEST_NUM], x[-TEST_NUM:]
y_train, y_test = y[:-TEST_NUM], y[-TEST_NUM:]
# 訓練
model = MLPRegressor()
model.fit(X=x_train, y=y_train)
# 預測
predict = model.predict(x_test)
print('Score = ', model.score(x_test, y_test))
# 繪製結果
plt.title(STOCK_FILE + ' stock price')
plt.xlabel('days')
plt.ylabel('price')
plt.plot(y_test,label='real')
plt.plot(predict,label='predict')
plt.legend(loc='upper left')
plt.show()

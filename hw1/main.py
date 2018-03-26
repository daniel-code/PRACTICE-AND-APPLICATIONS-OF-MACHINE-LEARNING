"""
線性迴歸主程式
python 3.5
Author:daniel-code
Date:2018.03.20
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import datasets

# mef
import hw1.my_LR as my_LR

# test sample number
TEST_NUM = 156

# dataset
boston = datasets.load_boston()
print(boston['feature_names'])
#print(boston['DESCR'])
x = boston.data
y = boston.target

#資料正規劃
x=np.array(x)
x=(x-x.mean())/x.std()

# splite dataset into train and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_NUM, random_state=0)

# sklearn model
model = LinearRegression()
model.fit(X=x_train, y=y_train)
y_predict = model.predict(X=x_test)
print('sklearn score = ',model.score(X=x_test,y=y_test))
print('sklearn coef = ',model.coef_)

#my_LR model
model2 = my_LR.my_LR()
model2.fit(x_train, y_train)
y_predict2 = model2.predict(x_test)
print('my_LR score = ',model2.score(X=x_test,y=y_test))
print('my_LR vs sklearn score = ',model2.score(X=x_test,y=y_predict))
print('my_LR coef = ',model2._thetas)

#plot predict
plt.figure(figsize=(12,6))
plt.title('LinearRegression')
plt.xlabel('#')
plt.ylabel('Price')
plt.plot(y_test, color='b',label='test data')
plt.plot(y_predict,color='r',label='sk predict')
plt.plot(y_predict2,color='g',label='my_LR predict')
plt.legend(loc='upper right')
plt.grid()
plt.show()


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

boston = load_boston()
x = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)
# 分析回归目标值的差异
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
y_train_T = y_train.reshape(len(y_train), -1)
y_test_T = y_test.reshape(len(y_test), -1)
print(y_train_T.shape)
print('the max target value is: ', np.max(boston.target))
print('the min target value is: ', np.min(boston.target))
print('the average target value is: ', np.mean(boston.target))
# 分别初始化对特征和目标值的标准化器
ss_x = StandardScaler()
ss_y = StandardScaler()
# 分别对训练和测试数据的特征以及目标值进行标准化处理
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)
y_train = ss_y.fit_transform(y_train_T)
y_test = ss_y.transform(y_test_T)
# 使用默认配置初始化线性回归器linearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_y_predict = lr.predict(x_test)

# 使用默认配置初始化新型回归器SGDRegression
sgdr = SGDRegressor()
sgdr.fit(x_train, y_train)
sgdr_y_predict = sgdr.predict(x_test)

# 评估
print('the value of default measurement of LinerRegression is: ', lr.score(x_test, y_test))
print('the value of R-squared if LinerRegression is: ', r2_score(y_test, lr_y_predict))
print('the mean squared error of LinearRegression is: ',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))
print('the mean absolute error of LinearRegression is: ',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))

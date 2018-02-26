from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge


X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]
X_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]
# 一次拟合
regressor = LinearRegression()
regressor.fit(X_train, y_train)

xx = np.linspace(0, 26, 100)  # 0-25 均匀取100个数
xx = xx.reshape(xx.shape[0], 1)
yy = regressor.predict(xx)
print('1次：')
print(regressor.score(X_test, y_test))
# 二次多项式拟合
poly2 = PolynomialFeatures(degree=2)  # 2次
X_train_poly2 = poly2.fit_transform(X_train)
regressor_poly2 = LinearRegression()
regressor_poly2.fit(X_train_poly2, y_train)
xx_poly2 = poly2.transform(xx)
yy_poly2 = regressor_poly2.predict(xx_poly2)
print('2次：')
X_test_poly2 = poly2.transform(X_test)
print(regressor_poly2.score(X_test_poly2, y_test))
# 4次多项式拟合
poly4 = PolynomialFeatures(degree=4)  # 4次
X_train_poly4 = poly4.fit_transform(X_train)
regressor_poly4 = LinearRegression()
regressor_poly4.fit(X_train_poly4, y_train)
xx_poly4 = poly4.transform(xx)
yy_poly4 = regressor_poly4.predict(xx_poly4)
print('4次：')
X_test_poly4 = poly4.transform(X_test)
print(regressor_poly4.score(X_test_poly4, y_test))


# L1范数正则化
lasso_poly4 = Lasso()
lasso_poly4.fit(X_train_poly4, y_train)  # 使用lasso对4次多项式特征进行拟合,输入参数已经是X_train_poly4了
yy_lasso_poly4 = lasso_poly4.predict(xx_poly4)
print('4次+L1：')
print(lasso_poly4.score(X_test_poly4, y_test))
# L2范数正则化
ridge_poly4 = Ridge()
ridge_poly4.fit(X_train_poly4, y_train)
yy_ridge_poly4 = ridge_poly4.predict(xx_poly4)
print('4次+L2：')
print(ridge_poly4.score(X_test_poly4, y_test))

# 作图
plt.scatter(X_train, y_train)
plt1, = plt.plot(xx, yy, label='Degree=1')
plt2, = plt.plot(xx, yy_poly2, label='Degree=2')
plt4, = plt.plot(xx, yy_poly4, label='Degree=4')
plt4_l1, = plt.plot(xx, yy_lasso_poly4, label='Degree=4,L1')
plt4_l2, = plt.plot(xx, yy_ridge_poly4, label='Degree=4,L2')
plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles=[plt1, plt2, plt4, plt4_l1, plt4_l2])
plt.show()

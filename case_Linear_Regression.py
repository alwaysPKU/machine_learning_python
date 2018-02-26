from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


X_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]

regressor = LinearRegression()
regressor.fit(X_train, y_train)

xx = np.linspace(0, 26, 100)  # 0-25 均匀取100个数
xx = xx.reshape(xx.shape[0], 1)
yy = regressor.predict(xx)

poly2 = PolynomialFeatures(degree=2)  # 2次
X_train_poly2 = poly2.fit_transform(X_train)
regressor_poly2 = LinearRegression()
regressor_poly2.fit(X_train_poly2, y_train)
xx_poly2 = poly2.transform(xx)
yy_poly2 = regressor_poly2.predict(xx_poly2)

poly4 = PolynomialFeatures(degree=4)  # 4次
X_train_poly4 = poly4.fit_transform(X_train)
regressor_poly4 = LinearRegression()
regressor_poly4.fit(X_train_poly4, y_train)
xx_poly4 = poly4.transform(xx)
yy_poly4 = regressor_poly4.predict(xx_poly4)

plt.scatter(X_train, y_train)
plt1, = plt.plot(xx, yy, label='Degree=1')
plt2, = plt.plot(xx, yy_poly2, label='Degree=2')
plt4, = plt.plot(xx, yy_poly4, label='Degree=4')
plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.legend(handles=[plt1, plt2, plt4])
plt.show()

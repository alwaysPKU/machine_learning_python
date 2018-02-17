from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

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
print('#####################LinearRegtession#####################')
print('the value of default measurement of LinearRegression is: ', lr.score(x_test, y_test))
print('the value of R-squared of LinerRegression is: ', r2_score(y_test, lr_y_predict))
print('the mean squared error of LinearRegression is: ',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))
print('the mean absolute error of LinearRegression is: ',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))

# 使用线性核函数配置的支持向量机进行回归训练，并且对测试样本进行预测。
linear_svr = SVR(kernel='linear')
linear_svr.fit(x_train, y_train)
linear_svr_y_predict = linear_svr.predict(x_test)

# 使用多项式核函数配置的支持向量机进行回归训练，并预测
poly_svr = SVR('poly')
poly_svr.fit(x_train, y_train)
poly_svr_y_predict = poly_svr.predict(x_test)

# 使用径向基核函数配置的支持向量机进行回归训练，并预测
rbf_svr = SVR('rbf')
rbf_svr.fit(x_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(x_test)

# 对三种核函数配置的SVM进行评估
print('#####################对三种核函数配置的SVM进行评估#####################')
print('=================linear SVR====================')
print('R-squared value of linear SVR is: ', linear_svr.score(x_test, y_test))
print('the mean squared error of linear SVR is: ',
      mean_squared_error(ss_y.inverse_transform(y_test),
                         ss_y.inverse_transform(linear_svr_y_predict)))
print('the mean absolute error of linear SVR is: ',
      mean_absolute_error(ss_y.inverse_transform(y_test),
                          ss_y.inverse_transform(linear_svr_y_predict)))
print('=================poly SVR====================')
print('R-squared value of linear SVR is: ', poly_svr.score(x_test, y_test))
print('the mean squared error of linear SVR is: ',
      mean_squared_error(ss_y.inverse_transform(y_test),
                         ss_y.inverse_transform(poly_svr_y_predict)))
print('the mean absolute error of linear SVR is: ',
      mean_absolute_error(ss_y.inverse_transform(y_test),
                          ss_y.inverse_transform(poly_svr_y_predict)))
print('=================rbf SVR====================')
print('R-squared value of linear SVR is: ', rbf_svr.score(x_test, y_test))
print('the mean squared error of linear SVR is: ',
      mean_squared_error(ss_y.inverse_transform(y_test),
                         ss_y.inverse_transform(rbf_svr_y_predict)))
print('the mean absolute error of linear SVR is: ',
      mean_absolute_error(ss_y.inverse_transform(y_test),
                          ss_y.inverse_transform(rbf_svr_y_predict)))

# 初始化K近邻回归器，并且调整配置，使得预测的方式为平均回归：weight='uniform'
uni_knr = KNeighborsRegressor(weights='uniform')
uni_knr.fit(x_train, y_train)
uni_knr_y_predict = uni_knr.predict(x_test)

# 初始化K近邻回归器，并且调整配置，是的预测的方式为根据距离加权回归：weight='distance'
dis_knr = KNeighborsRegressor(weights='distance')
dis_knr.fit(x_train, y_train)
dis_knr_y_predict = dis_knr.predict(x_test)

# 评估
print('#####################对两种不同配置的K近邻模型进行评估#####################')
print('=================uniform-weighted====================')
print('R-squared value of uniform-weighted KNeighorRegression: ', uni_knr.score(x_test, y_test))
print('the mean squared error if uniform-weighted KNeighorRegression: ',
      mean_squared_error(ss_y.inverse_transform(y_test),
                         ss_y.inverse_transform(uni_knr_y_predict)))
print('the mean absolute error of uniform-weighted KNeighorRegression: ',
      mean_absolute_error(ss_y.inverse_transform(y_test),
                          ss_y.inverse_transform(uni_knr_y_predict)))

print('=================distance-weighted====================')
print('R-squared value of distance-weighted KNeighorRegression: ', dis_knr.score(x_test, y_test))
print('the mean squared error if distance-weighted KNeighorRegression: ',
      mean_squared_error(ss_y.inverse_transform(y_test),
                         ss_y.inverse_transform(dis_knr_y_predict)))
print('the mean absolute error of distance-weighted KNeighorRegression: ',
      mean_absolute_error(ss_y.inverse_transform(y_test),
                          ss_y.inverse_transform(dis_knr_y_predict)))

# 回归树'不需要对特征标准化和量化'
dtr = DecisionTreeRegressor()
dtr.fit(x_train, y_train)
dtr_y_predict = dtr.predict(x_test)
# 评估
print('#####################回归树DecisionTreeRegressor#####################')
print('R-squared value of DecisionTreeRegressor: ', dtr.score(x_test, y_test))
print('the mean squared error of DecisionTreeRegressor: ',
      mean_squared_error(ss_y.inverse_transform(y_test),
                         ss_y.inverse_transform(dtr_y_predict)))
print('the mean absolute error of DecisionTreeRegressor: ',
      mean_absolute_error(ss_y.inverse_transform(y_test),
                          ss_y.inverse_transform(dtr_y_predict)))

# 集成模型
# 随机森林
rft = RandomForestRegressor()
rft.fit(x_train, y_train)
rft_y_predict = rft.predict(x_test)
# 极端随机森林
etr = ExtraTreesRegressor()
etr.fit(x_train, y_train)
etr_y_predict = etr.predict(x_test)
# 梯度提升GBR
gbr = GradientBoostingRegressor()
gbr.fit(x_train, y_train)
gbr_y_predict = gbr.predict(x_test)

# 评估
print('#####################集成模型#####################')
print('=================RandomForestRegressor====================')
print('R-squared value of RandomForestRegressor: ', rft.score(x_test, y_test))
print('the mean squared error of RandomForestRegressor: ',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rft_y_predict)))
print('the mean absolute error of RandomForestRegressor: ',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rft_y_predict)))
print('=================ExtraTreeRegressor====================')
print('R-squared value of ExtraTreeRegressor: ', etr.score(x_test, y_test))
print('the mean squared error of ExtraTreeRegressor: ',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_predict)))
print('the mean absolute error of ExtraTreeRegressor: ',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_predict)))
# 利用训练好的极端回归树模型，输出每种特征对预测目标的贡献度：
# print(np.sort(zip(etr.feature_importances_, boston.feature_names), axis=0))

print('=================GradientBoostingRegressor====================')
print('R-squared value of GradientBoostingRegressor: ', gbr.score(x_test, y_test))
print('the mean squared error of GradientBoostingRegressor: ',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_predict)))
print('the mean absolute error of GradientBoostingRegressor: ',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_predict)))

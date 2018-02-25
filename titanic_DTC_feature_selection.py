import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import feature_selection
from sklearn.model_selection import cross_val_score
import pylab as pl

titanic = pd.read_csv('titanic.txt')
# 分离数据特征与预测目标
y = titanic['survived']
X = titanic.drop(['row.names', 'name', 'survived'], axis=1)

# 对缺失数据进行填充
X['age'].fillna(X['age'].mean(), inplace=True)
X.fillna('UNKNOWN', inplace=True)

# 分割数据，采样25%测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)

# 特征向量化
vec = DictVectorizer()
X_train = vec.fit_transform(X_train.to_dict(orient='record'))
X_test = vec.transform(X_test.to_dict(orient='record'))

print(len(vec.feature_names_))
print(vec.feature_names_)

# 使用决策树模型依靠所有特征进行预测,用熵参数
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train, y_train)
print(dt.score(X_test, y_test))
print('==========================')

# 特征筛选
fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=50)  # 筛选前50%的特征
X_train_fs = fs.fit_transform(X_train, y_train)
dt.fit(X_train_fs, y_train)
X_test_fs = fs.transform(X_test)
print(dt.score(X_test_fs, y_test))

# 交叉验证，按照固定间隔的百分比筛选特征，并作图展示性能岁筛选比例的变化
percentiles = range(1, 100, 2)
results = []
for i in percentiles:
    fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=i)
    X_train_fs = fs.fit_transform(X_train, y_train)
    scores = cross_val_score(dt, X_train_fs, y_train, cv=5)
    results = np.append(results, scores.mean())
print(results)
# 找到最佳性能的特征筛选百分比
opt = np.where(results == results.max())[0][0]
print('optimal number of features %d' % percentiles[opt])

pl.plot(percentiles, results)
pl.xlabel('percentiles of features')
pl.ylabel('accuracy')
pl.show()

fs = feature_selection.SelectPercentile(feature_selection.chi2, percentile=opt)
X_train_fs = fs.fit_transform(X_train, y_train)
dt.fit(X_train_fs, y_train)
X_test_fs = fs.transform(X_test)
print(dt.score(X_test_fs, y_test))


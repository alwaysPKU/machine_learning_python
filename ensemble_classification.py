import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report


titanic = pd.read_csv('titanic.txt')
x = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']
x['age'].fillna(x['age'].mean(), inplace=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)
# 对类别类型特征进行转化，成为特征向量
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
vec = DictVectorizer(sparse=False)
x_train = vec.fit_transform(x_train.to_dict(orient='record'))
x_test = vec.transform(x_test.to_dict(orient='record'))
# 使用单一决策树进行模型训练以及预测分析
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
dtc_y_predict = dtc.predict(x_test)

# 使用随机森林分类器进行集成模型的训练以及预测分析
rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)
rfc_y_predict = rfc.predict(x_test)

# 使用梯度提升决策树进行集成模型的训练集预测
gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)
gbc_y_predict = gbc.predict(x_test)

# 评估
print('the accuracy of decision tress is:', dtc.score(x_test, y_test))
print(classification_report(dtc_y_predict, y_test))

print('the accuracy of random forest classifier is:', rfc.score(x_test, y_test))
print(classification_report(rfc_y_predict, y_test))

print('the accuracy of gradient tree boosting is:', gbc.score(x_test, y_test))
print(classification_report(gbc_y_predict, y_test))

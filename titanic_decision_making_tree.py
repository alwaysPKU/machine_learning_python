import pandas as pd
from sklearn.model_selection import train_test_split
# 特征转换器
from sklearn.feature_extraction import DictVectorizer
# 决策树
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


# titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
titanic = pd.read_csv('titanic.txt')
# print(titanic.info())
x = titanic[['pclass', 'age', 'sex']]
print(x.info())
y = titanic['survived']
# age 空值用mean填充
x['age'].fillna(x['age'].mean(), inplace=True)
print(x.info())
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)
# 特征转换,凡是类别性的特征单独剥离出来
vec = DictVectorizer(sparse=False)
x_train = vec.fit_transform(x_train.to_dict(orient='record'))
# print(vec.feature_names_)
x_test = vec.transform(x_test.to_dict(orient='record'))
print(vec.feature_names_)
# 决策树
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
y_predict = dtc.predict(x_test)
# 评估
print(dtc.score(x_test, y_test))
print(classification_report(y_predict, y_test, target_names=['died', 'survived']))

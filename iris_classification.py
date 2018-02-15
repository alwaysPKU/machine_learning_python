# 导入iris数据集
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# 标准化
from sklearn.preprocessing import StandardScaler
# K临近分类器
from sklearn.neighbors import KNeighborsClassifier
# 评估
from sklearn.metrics import classification_report


iris = load_iris()
# print(iris.DESCR)
x_train, x_test, y_train, y_test = \
    train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)
# 标准化
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
# 预测
knc = KNeighborsClassifier()
knc.fit(x_train, y_train)
y_predict = knc.predict(x_test)
# 评估
print('the accuracy of k-nearest neighbor classifier is:', knc.score(x_test, y_test))
print(classification_report(y_test, y_predict, target_names=iris.target_names))

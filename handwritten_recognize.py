# 从sklearn.datasets里导入手写体数字加载器
import numpy as np
from sklearn.datasets import load_digits
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
# 数据标准化模块
from sklearn.preprocessing import StandardScaler
# 从 sklearn.svm里导入基于线性假设的支持向量机分类器
# LinearSVC/Linear Support Vector Classification.
from sklearn.svm import LinearSVC
# 依然使用sklearn.metrics 里面的 classification_report 模块对预测加过分析
from sklearn.metrics import classification_report

digits = load_digits()
print(digits.data.shape)
x_train, x_test, y_train, y_test = \
    train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)
print('训练集', y_train.shape)
print('测试集', y_test.shape)
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)

# 训练及预测
lsvc = LinearSVC()
lsvc.fit(x_train, y_train)
y_predict = lsvc.predict(x_test)

# 使用模型自带的评估函数进行评估
print('the accuracy of liner SVC is: ', lsvc.score(x_test, y_test))
# 详细分析：
print(classification_report(y_test, y_predict, target_names=digits.target_names.astype(str)))


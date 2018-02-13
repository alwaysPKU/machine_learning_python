# 从sklearn.datasets 里导入新闻数据抓取器fetch_20newsgroups
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
# 从sklearn.feature_extraction.text导入基于文本特征向量转化模块
from sklearn.feature_extraction.text import CountVectorizer
# 导入朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB
# 报告
from sklearn.metrics import classification_report


# 与之前预存的数据不同，fetch_20newsgroups需要从互联网上即时下载
news = fetch_20newsgroups(subset='all')
# 检查数据规模和细节
print(len(news.data))
print(news.data[0])
print(news.data[1])
x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)
vec = CountVectorizer()
x_train = vec.fit_transform(x_train)
x_test = vec.transform(x_test)
mnb = MultinomialNB()
mnb.fit(x_train, y_train)
y_predict = mnb.predict(x_test)

print('the accuracy of naive Bayes classifier is:', mnb.score(x_test, y_test))
print(classification_report(y_test, y_predict, target_names=news.target_names))


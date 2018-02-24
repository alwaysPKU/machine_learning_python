# 从sklearn.datasets 里导入新闻数据抓取器fetch_20newsgroups
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
# 从sklearn.feature_extraction.text导入基于文本特征向量转化模块
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# 导入朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB
# 报告
from sklearn.metrics import classification_report


# 与之前预存的数据不同，fetch_20newsgroups需要从互联网上即时下载
news = fetch_20newsgroups(subset='all')
# 检查数据规模和细节
print('============================================================')
print(len(news.data))
print('============================================================')
print(news.data[0])
print('============================================================')
print(news.target[0])
print('============================================================')
print(news.data[1])
print('============================================================')
print(news.target[1])
print('============================================================')
x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)
# 1. 使用CountVectorizer 并且不去掉停用词的条件下，对文本进行量化，朴素贝叶斯分类性能
vec = CountVectorizer()
X_train = vec.fit_transform(x_train)
X_test = vec.transform(x_test)
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_predict = mnb.predict(X_test)
print('#########使用CountVectorizer 并且不去掉停用词的条件下，对文本进行量化，朴素贝叶斯分类性能############')
print('the accuracy of naive Bayes classifier is:', mnb.score(X_test, y_test))
print('============================================================')
print(classification_report(y_test, y_predict, target_names=news.target_names))

# 2. 使用TfidfVectorizer并且不去掉停用词的条件下，对文本向量化，朴素贝叶斯分类性能
tfidf_vec = TfidfVectorizer()
X_tfdif_train = tfidf_vec.fit_transform(x_train)
X_tfdif_test = tfidf_vec.transform(x_test)
mnb_tifdif = MultinomialNB()
mnb_tifdif.fit(X_tfdif_train, y_train)
y_tifdif_predict = mnb_tifdif.predict(X_tfdif_test)
print('#########使用TfidfVectorizer并且不去掉停用词的条件下，对文本向量化，朴素贝叶斯分类性能############')
print('the accuracy of naive Bayes classifier is:', mnb.score(X_tfdif_test, y_test))
print('============================================================')
print(classification_report(y_test, y_tifdif_predict, target_names=news.target_names))

# 3. 使用CountVectorizer与 并且去掉停用词的条件下，对文本进行量化，朴素贝叶斯分类性能

print('#########使用CountVectorizer与TfidfVectorizer并且去掉停用词的条件下，对文本进行量化，朴素贝叶斯分类性能############')
count_filter_vec, tifdif_filter_vec = \
    CountVectorizer(analyzer='word', stop_words='english'), \
    TfidfVectorizer(analyzer='word', stop_words='english')

X_count_filter_train = count_filter_vec.fit_transform(x_train)
X_count_filter_test = count_filter_vec.transform(x_test)

X_tifdif_filter_train = tifdif_filter_vec.fit_transform(x_train)
X_tifdif_filter_test = tifdif_filter_vec.transform(x_test)

mnb_count_filter = MultinomialNB()
mnb_count_filter.fit(X_count_filter_train, y_train)
y_count_filter_predict = mnb_count_filter.predict(X_count_filter_test)
print('============================================================')
print('the accuracy of classifying 20newsgroups using naive bayes(countVectorizer by filtering stopwords): ',
      mnb_count_filter.score(X_count_filter_test, y_test))
print('============================================================')
print(classification_report(y_test, y_count_filter_predict, target_names=news.target_names))
print('============================================================')

mnb_tifdif_filter = MultinomialNB()
mnb_tifdif_filter.fit(X_tifdif_filter_train, y_train)
y_tifdif_filter_predict = mnb_tifdif_filter.predict(X_tifdif_filter_test)
print('============================================================')
print('the accuracy of classifying 20newsgroups using naive bayes(TfidfVectorizer by filtering stopwords:',
      mnb_tifdif_filter.score(X_tifdif_filter_test, y_test))
print('============================================================')
print(classification_report(y_test, y_tifdif_filter_predict, target_names=news.target_names))
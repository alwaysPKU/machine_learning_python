from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
# from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV

news = fetch_20newsgroups(subset='all')
X_train, X_test, y_train, y_test = train_test_split(news.data[:3000], news.target[:3000], test_size=0.25, random_state=33)

# 使用pipeline简化系统搭建流程，将文本抽取与分类器模型串联起来
clf = Pipeline([('vect', TfidfVectorizer(stop_words='english', analyzer='word')), ('svc', SVC())])

# 这里需要实验的2个超参数的个数分别是4，3.12种组合
parameters = {'svc__gamma': np.logspace(-2, 1, 4), 'svc__C': np.logspace(-1, 1, 3)}

# 将12组参数组合以及初始化的Pipline包括3折交叉验证的要求全部告知GridSearchCV，12*3 = 36次计算
# 可以加入参数n_jobs=-1使用计算机全部CPU！！！！！！！！！！！！！！！！！！！1
gs = GridSearchCV(clf, parameters, verbose=2, refit=True, cv=3)

# 执行网格搜索
gs.fit(X_train, y_train)
gs.best_params_, gs.best_score_
# refit=True即保存了效果最好的参数组合。
print(gs.score(X_test, y_test))

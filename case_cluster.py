import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


digits_train = pd.read_csv\
    ('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)
digits_test = pd.read_csv\
    ('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header=None)

print(digits_train.shape)
print(digits_test.shape)

x_train = digits_train[np.arange(64)]
y_train = digits_train[64]

x_test = digits_test[np.arange(64)]
y_test = digits_test[64]

# 初始化kmeans，设置k=10
kmeans = KMeans(n_clusters=10)
kmeans.fit(x_train)
y_predict = kmeans.predict(x_test)
# 评估
# 1. 如果被评估数据本身有正确类别信息，ARI: adjusted rand index
print(metrics.adjusted_rand_score(y_test, y_predict))
# 2. 如果被评估数据没有所属类别，用轮廓系数(silhouette coefficient)，系数值越大分类越好
# 分割出3*2=6个子图，并在1号子图作图
plt.subplot(3, 2, 1)
# 初始化原始数据点
x1 = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
x2 = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
X_tmp = np.array([[1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9], [1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3]])
X = X_tmp.transpose()
# x2 = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])

print(X)
# 在1号子图做出原始
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('Instance')
plt.scatter(x1, x2)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+ ']

clusters = [2, 3, 4, 5, 8]
subplot_counter = 1
sc_scores = []
for t in clusters:
    subplot_counter += 1
    plt.subplot(3, 2, subplot_counter)
    kmeans_model = KMeans(n_clusters=t).fit(X)
    for i, l in enumerate(kmeans_model.labels_):
        plt.plot(x1[i], x2[i], color=colors[l], marker=markers[1], ls='None')
        plt.xlim([0, 10])
        plt.ylim([0, 10])
        sc_score = silhouette_score(X, kmeans_model.labels_, metric='euclidean')
        sc_scores.append(sc_score)
        plt.title('K=%s, silhouette coefficient = %0.03f'%(t, sc_score))
        plt.figure()
        plt.plot(clusters, sc_scores, '*-')
        plt.xlabel('Number of Cluster')
        plt.ylabel('Silhouette Coefficient Score')
        plt.show()


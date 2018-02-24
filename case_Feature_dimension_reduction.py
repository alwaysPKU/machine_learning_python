import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


digits_train = pd.read_csv\
    ('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)
digits_test = pd.read_csv\
    ('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header=None)

X_digits = digits_train[np.arange(64)]
y_digits = digits_train[64]

# 初始化
# 1. 二维
estimator = PCA(n_components=2)
X_pca = estimator.fit_transform(X_digits)
# 2. 三维
estimator_3 = PCA(n_components=3)
X_pca_3 = estimator_3.fit_transform(X_digits)


# 显示10类手写体数字图片经PCA压缩后的2维空间分布
def plot_pca_scatter():
    colors = ['black', 'blue', 'purple', 'yellow', 'magenta', 'red', 'lime', 'cyan', 'orange', 'gray']
    for i in range(len(colors)):  # 按顺序一个颜色一个颜色的循环
        px = X_pca[:, 0][y_digits.as_matrix() == i]  # i颜色第一维数据
        py = X_pca[:, 1][y_digits.as_matrix() == i]  # 颜色第二维数据
        plt.scatter(px, py, c=colors[i])
    plt.legend(np.arange(0, 10).astype(str))  # 显示图例
    plt.xlabel('first principal components')
    plt.ylabel('second principal components')
    plt.show()


def plot_pca_scatter3():
    colors = ['black', 'blue', 'purple', 'yellow', 'magenta', 'red', 'lime', 'cyan', 'orange', 'gray']
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(len(colors)):  # 按顺序一个颜色一个颜色的循环
        px = X_pca_3[:, 0][y_digits.as_matrix() == i]  # 第一维数据
        py = X_pca_3[:, 1][y_digits.as_matrix() == i]  # 第二维数据
        pz = X_pca_3[:, 2][y_digits.as_matrix() == i]  # 第三维数据
        ax.scatter(px, py, pz, c=colors[i])
    ax.legend(np.arange(0, 10).astype(str))
    ax.set_xlabel('first principal components')
    ax.set_ylabel('second principal components')
    ax.set_zlabel('third principal components')
    plt.show()
# plot_pca_scatter()
plot_pca_scatter3()
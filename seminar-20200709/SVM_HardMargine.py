from sklearn import svm
from sklearn.datasets import make_classification
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import mglearn

cm_bright = ListedColormap(['#FF0000', '#0000FF'])

X, y = make_classification(
    n_samples = 100,
    n_features = 2,
    n_informative = 2,
    n_redundant = 0,
    n_classes = 2,
    n_clusters_per_class = 1,
    class_sep = 2.0,
    shift = None,
    random_state = 5
)

# 線形分離
clf = svm.SVC(kernel = 'linear', C = 1000)
clf.fit(X, y)

# マージン境界(上限, 下限)と決定境界の計算
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
margin_down = yy - np.sqrt(1 + a ** 2) * margin
margin_up = yy + np.sqrt(1 + a ** 2) * margin

# 描画処理
fig = plt.figure()
plt.scatter(X[:,0], X[:,1], s=25, c=y, cmap=cm_bright)
plt.plot(xx, margin_down, linestyle = 'dashed', color = 'black')
plt.plot(xx, margin_up, linestyle = 'dashed', color = 'black')
plt.plot(1.0, -3.0, marker='o', markersize=5, color = 'green')
plt.plot(1.0, -2.5, marker='o', markersize=5, color = 'green')

mglearn.plots.plot_2d_separator(clf, X)
fig.savefig("img.png")

# テスト
testX = np.array([[1.0, -3.0], [1.0, -2.5]])
judge = clf.predict(testX)

# 青 : クラス1, 赤 : クラス0
print('{0} is class {1}'.format(tuple(testX[0]), judge[0]))
print('{0} is class {1}'.format(tuple(testX[1]), judge[1]))
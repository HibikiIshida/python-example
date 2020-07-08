from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.datasets import make_classification
from sklearn import svm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import mglearn
import numpy as np

cm_bright = ListedColormap(['#FF0000', '#0000FF'])

X, y = make_classification(
    n_samples = 100,
    n_features = 2,
    n_informative = 2,
    n_redundant = 0,
    n_classes = 2,
    n_clusters_per_class = 1,
    class_sep = 0.4,
    shift = None,
    random_state = 5
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

clf = svm.SVC(kernel = 'linear', C = 10000)
clf.fit(X_train, y_train)

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

mglearn.plots.plot_2d_separator(clf, X)
fig.savefig("img2.png")

# トレーニングデータに対する性能評価
X_train_pred = clf.predict(X_train)
print('confusion = \n %s' % confusion_matrix(y_train, X_train_pred))
print(classification_report(y_train, X_train_pred))

# テストデータに対する性能評価
X_test_pred = clf.predict(X_test)
print('confusion = \n %s' % confusion_matrix(y_test, X_test_pred))
print(classification_report(y_test, X_test_pred))
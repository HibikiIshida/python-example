from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


X, y = make_blobs(
    n_samples = 600,
    n_features = 2,
    centers = 4,
    cluster_std = 1.0,
    random_state = 2
)

kmeans = KMeans(n_clusters = 4)
y_train_est = kmeans.fit_predict(X)

fig = plt.figure()
plt.scatter(X[:,0], X[:,1], s=25, c=y_train_est)

fig.savefig("img.png")
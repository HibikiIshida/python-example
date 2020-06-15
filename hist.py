from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

num = 10000
N = 10
mean, std = 2, 0.5
mu = np.zeros(num)

for i in range(num) :
    mu[i] = np.mean(norm.rvs(loc=mean, scale=std, size=N))

plt.hist(mu, bins=20, range=(1.5, 2.5), ec='black')
plt.show()
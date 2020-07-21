from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df_all = pd.read_csv("https://sites.google.com/site/datasciencehiro/datasets/Wholesale_customers_data.csv")
df_l = df_all.drop(['Channel', 'Region'], axis=1)

# 分析するデータ
df_l.head()

cstmr_data = np.array([
    df_l['Fresh'].values,
    df_l['Milk'].values,
    df_l['Grocery'].values,
    df_l['Frozen'].values,
    df_l['Detergents_Paper'].values,
    df_l['Delicassen'].values
])

cstmr_data = cstmr_data.T

# クラスタリング
clstr = KMeans(n_clusters=4).fit_predict(cstmr_data)

# クラスタリング結果を分析データに追加
df_l['cluster_id'] = clstr

# Matplotlibで積み上げ棒グラフにして出力
clusterinfo = pd.DataFrame()
for i in range(4):
    clusterinfo['cluster' + str(i)] = df_l[df_l['cluster_id'] == i].mean()
    print("・クラスタ番号{0}".format(i))
    print(df_l[df_l['cluster_id'] == i].mean())
    print("")

clusterinfo = clusterinfo.drop('cluster_id')
my_plot = clusterinfo.T.plot(kind='bar', stacked=True, title="Mean Value of 4 Clusters")
my_plot.set_xticklabels(my_plot.xaxis.get_majorticklabels(), rotation=0)
plt.savefig('image.png')

# GroceryとDetergents_Paperの相関関係
cstmr_data = np.array([
    df_l['Grocery'].tolist(),
    df_l['Detergents_Paper'].tolist()
])

fig = plt.figure()
cstmr_data = cstmr_data.T
clstr = KMeans(n_clusters=3).fit_predict(cstmr_data)
plt.scatter(cstmr_data[:, 0], cstmr_data[:, 1], c=clstr, edgecolors='k')
plt.savefig('image2.png')
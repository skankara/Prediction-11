import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import style
style.use("ggplot")

# x = [1,5,1.5,8,1,9]
# y = [2,8,1.8,8,0.6,11]
# plt.scatter(x, y)
# plt.show()
# X = np.array([[1, 2],
#               [5, 8],
#               [1.5, 1.8],
#               [8, 8],
#               [1, 0.6],
#               [9, 11]])
# kmeans = KMeans(n_clusters=2)
# kmeans.fit(X)
# centroids = kmeans.cluster_centers_
# labels = kmeans.labels_
# print(centroids)
# print(labels)
# colors = ["g.","r."]
# for i in range(len(X)):
#     print("coord",X[i], "label : ", labels[i])
#     plt.plot(X[i][0], X[i][1],colors[labels[i]], markersize = 10)
# plt.scatter(centroids[:,0],centroids[:,1], marker = "x", s =150, linewidths=5, zorder =10)
# plt.show()

center = [[1,1],[5,5]]
X, Y = make_blobs(n_samples = 100, centers =center, cluster_std=1)
print(X.dtypes)
plt.scatter(X[:,0],X[:,1])
plt.show()
ms = MeanShift()
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
n_clusters_ = len(np.unique(labels))
print("clusters : ",n_clusters_)
colors = 10*['r.','c.','k.','g.','b.','y.','m.']
print(colors)
print(labels)
for i in range(len(X)):
    print(X)
    #print("coord",X[i], "label : ", labels[i])
    plt.plot(X[i][0], X[i][1],colors[labels[i]], markersize = 10)
plt.scatter(cluster_centers[:,0],cluster_centers[:,1], marker = "x", s =150, linewidths=5, zorder =10)
plt.show()


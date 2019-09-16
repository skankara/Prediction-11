import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift, Birch, DBSCAN

deliveries = pd.read_csv('ipl/deliveries_new.csv')
data = deliveries[['match_id','over','batsman','bowler','batsman_runs',	'extra_runs']]
data.to_csv('ipl/data.csv',index=False)

runsScored = []
oversPlayed = []
for j in data["bowler"].unique():
    r = data["batsman_runs"][data["bowler"] == j].sum() + data["extra_runs"][data["bowler"] == j].sum()
    b = data["over"][data["bowler"] == j].count()/6
    runsScored.append(r)
    oversPlayed.append(b)

All_bowlers = pd.DataFrame({"Player": data["bowler"].unique()})
All_bowlers["Runs"] = runsScored
All_bowlers["Overs"] = oversPlayed
All_bowlers["Economy_Rate"] = (All_bowlers["Runs"] / All_bowlers["Overs"]).round(2)

# (All_bowlers["Economy_Rate"].to_frame()).hist(color="orange")
# plt.title("Histogram of Bowler Economy Rate")
# plt.xlabel("Economy Rate")
# plt.ylabel("No. of Bowlers")
# plt.show()
# All_bowlers.to_csv('ipl/All_bowlers.csv',index=False)

#Filtering Bowlers
All_bowlers = All_bowlers[(All_bowlers["Economy_Rate"] >6) & (All_bowlers["Economy_Rate"] <10)]
(All_bowlers["Economy_Rate"].to_frame()).hist(color="orange")
plt.title("Histogram of Bowler Economy Rate")
plt.xlabel("Economy Rate")
plt.ylabel("No. of Bowlers")
plt.show()

#Scatter Plot
fig = plt.figure(figsize=(10,5))
plt.xlabel("Economy Rate")
plt.ylabel("Runs")
plt.scatter(All_bowlers["Economy_Rate"], All_bowlers["Runs"], color = "orange")
plt.title("Bowler : Economy Rate Vs Runs")
plt.show()

kmeans = KMeans(n_clusters=5)
kmeans.fit(All_bowlers[["Economy_Rate", "Runs"]])
centroids = kmeans.cluster_centers_
All_bowlers["cluster"] = kmeans.labels_
kmeansplot = plt.figure(figsize=(10, 5))
colors = ["red", "orange", "yellow", "blue", "green", "indigo", "black"]
for r in range(1, 6):
    clustered_Bowler = All_bowlers[All_bowlers["cluster"] == r]
    plt.scatter(clustered_Bowler["Economy_Rate"], clustered_Bowler["Runs"], color=colors[r - 1])
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
plt.title("IPL Bowlers clustering using K-Means", fontsize=16)
plt.xlabel("Economy Rate of the Bowler")
plt.ylabel("Runs given by the Bowler")
plt.show()

center = [[1, 1], [5, 5]]
X = np.asarray(All_bowlers[["Economy_Rate", "Runs"]])
#print(X)
# X, Y = make_blobs(n_samples = 100, centers =center, cluster_std=1)
# plt.scatter(X[:,0],X[:,1])
# plt.show()
#print(X.dtypes)
ms = MeanShift()
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
n_clusters_ = len(np.unique(labels))
#print("clusters : ", n_clusters_)
plt.figure(figsize=(10, 5))
colors = 10*['r.', 'c.', 'k.', 'g.', 'b.', 'y.', 'm.']
# print(colors)
# print(labels)
for i in range(len(X)):
    plt.plot(X[i][0], X[i][1],colors[labels[i]], markersize = 10)
plt.scatter(cluster_centers[:,0],cluster_centers[:,1], marker = "x", s =150, linewidths=5, zorder =10)
plt.title("IPL Bowlers clustering using MeanShift", fontsize=16)
plt.xlabel("Economy Rate", fontsize=14)
plt.ylabel("Runs", fontsize=14)
plt.show()


#BIRCH
colors = 10*[ 'b.', 'y.', 'm.','r.', 'c.', 'k.', 'g.']
birch = Birch(branching_factor=50, n_clusters=5, threshold=0.5, compute_labels=True)

birch.fit(X)
plt.figure(figsize=(10, 5))
#cluster_centers = birch.subcluster_centers_
for i in range(len(X)):
    plt.plot(X[i][0], X[i][1],colors[labels[i]], markersize = 10)
#plt.scatter(cluster_centers[:,0],cluster_centers[:,1], marker = "x", s =150, linewidths=5, zorder =10)
plt.title("IPL Bowlers clustering using BIRCH", fontsize=16)
plt.xlabel("Economy Rate", fontsize=14)
plt.ylabel("Runs", fontsize=14)
plt.show()


#DBSCAN
colors = 10*[  'm.','r.','b.', 'y.', 'c.', 'k.', 'g.']
db = DBSCAN(eps=5, min_samples=5, metric='euclidean', algorithm='auto')
db.fit(X)
plt.figure(figsize=(10,5))
for i in range(len(X)):
    plt.plot(X[i][0], X[i][1],colors[labels[i]], markersize = 10)
#plt.scatter(cluster_centers[:,0],cluster_centers[:,1], marker = "x", s =150, linewidths=5, zorder =10)
plt.title("IPL Bowlers clustering using DBSCAN", fontsize=16)
plt.xlabel("Economy Rate", fontsize=14)
plt.ylabel("Runs", fontsize=14)
plt.show()

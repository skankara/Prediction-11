import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift, Birch, DBSCAN


#batsman = []
#bowler = []
deliveries = pd.read_csv('ipl/deliveries_new.csv')
data = deliveries[['match_id','over','batsman','bowler','batsman_runs',	'extra_runs']]
#print(data.dtypes)
#data.to_csv('ipl/data.csv',index=False)
data.isnull().sum()
#data["extra_runs"] = pd.to_numeric(data["extra_runs"], errors="coerce")
#data[["extra_runs","batsman_runs"]] = data[["extra_runs","batsman_runs"]].apply(pd.to_numeric)
#data[["extra_runs"]] = data[["extra_runs"]].fillna(0)
#data["batsman_runs"] = pd.to_numeric(data["batsman_runs"], errors="coerce")
#data[["batsman_runs"]] = data[["batsman_runs"]].fillna(0)
#data.to_csv('ipl/data.csv',index=False)
#print(data.dtypes)

runsScored = []
ballsPlayed = []
for i in data["batsman"].unique():
    r = data["batsman_runs"][data["batsman"] == i].sum()
    b = data["match_id"][data["batsman"] == i].count()
    runsScored.append(r)
    ballsPlayed.append(b)

All_batsman = pd.DataFrame({"Player":data["batsman"].unique()})
All_batsman["Runs"] = runsScored
All_batsman["Balls"] = ballsPlayed
All_batsman["Strike_Rate"] = ((All_batsman["Runs"] / All_batsman["Balls"])*100).round(2)

# (All_batsman["Strike_Rate"].to_frame()).hist(color="orange")
# plt.title("Histogram of Batsman Strike Rate")
# plt.xlabel("Strike Rate")
# plt.ylabel("No. of Batsman")
# plt.show()
#All_batsman.to_csv('ipl/All_batsman.csv',index=False)

#Filtering Batsman

#All_batsman = All_batsman[All_batsman["Strike_Rate"] > 60]
#All_batsman = All_batsman[All_batsman["Strike_Rate"] < 150]
#All_batsman = All_batsman[(All_batsman["Runs"]*6/All_batsman["Balls"]) > 6]
All_batsman = All_batsman[All_batsman["Balls"] > 250]
fig = (All_batsman["Strike_Rate"].to_frame()).hist(color="orange")
#plt.figure(figsize=(10,5))
plt.title("Histogram of Batsman Strike Rate")
plt.xlabel("Strike Rate")
plt.ylabel("No. of Batsman")
plt.show()

#Scatter Plot
plt.figure(figsize=(10,5))
plt.xlabel("Strike Rate")
plt.ylabel("Runs")
plt.scatter(All_batsman["Strike_Rate"], All_batsman["Runs"], color = "orange")
plt.title("Batsmen : Strike Rate Vs Runs")
plt.show()

#K-MEANS
kmeans = KMeans(n_clusters=4)
kmeans.fit(All_batsman[["Strike_Rate", "Runs"]])
centroids = kmeans.cluster_centers_
All_batsman["cluster"] = kmeans.labels_
plt.figure(figsize=(10, 5))
colors = ["red", "orange", "yellow", "blue", "green", "indigo", "black"]
for r in range(1, 6):
    clustered_Batsmen = All_batsman[All_batsman["cluster"] == r]
    plt.scatter(clustered_Batsmen["Strike_Rate"], clustered_Batsmen["Runs"], color=colors[r - 1])
plt.title("IPL Batsmen clustering using K-Means", fontsize=16)
plt.xlabel("Strike Rate", fontsize=14)
plt.ylabel("Runs", fontsize=14)
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
plt.show()

#Mean Shift
#All_batsman.to_csv('ipl/All_batsman_cluster.csv',index=False)
center = [[1, 1], [5, 5]]
X = np.asarray(All_batsman[["Strike_Rate", "Runs"]])
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
plt.title("IPL Batsmen clustering using MeanShift", fontsize=16)
plt.xlabel("Strike Rate", fontsize=14)
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
plt.title("IPL Batsmen clustering using BIRCH", fontsize=16)
plt.xlabel("Strike Rate", fontsize=14)
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
plt.title("IPL Batsmen clustering using DBSCAN", fontsize=16)
plt.xlabel("Strike Rate", fontsize=14)
plt.ylabel("Runs", fontsize=14)
plt.show()

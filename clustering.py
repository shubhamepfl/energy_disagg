from sklearn.cluster import KMeans
def clustering(data, n_clusters):
    kmeans= KMeans(init='k-means++', n_clusters=n_clusters)
    kmeans.fit(data)
    return kmeans.cluster_centers_, kmeans.labels_
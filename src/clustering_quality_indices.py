import numpy as np


def dist_intra_in_cluster(dict_dist, C):
    """Computes the distance in a cluster."""
    if len(C) == 1:
        return 0
    temp = []
    for i in range(len(C) - 1):
        for j in range(i + 1, len(C)):
            temp.append(dict_dist[(tuple(sorted((C[i], C[j]))))])
    return np.max(temp)
    # return np.mean(dist_tmp)


def dist_between_clusters(dict_dist, C1, C2):
    """Computes the distance between two clusters."""
    temp = []
    for i in C1:
        for j in C2:
            temp.append(dict_dist[tuple(sorted((i, j)))])
    return np.min(temp)
    # return np.mean(temp)


def smallest_dist_between_clusters(dict_dist, dict_clusters):
    clusters = list(dict_clusters.keys())
    list_dist = []
    for i in range(len(clusters) - 1):
        C1 = clusters[i]
        for j in range(i + 1, len(clusters)):
            C2 = clusters[j]
            list_dist.append(
                dist_between_clusters(dict_dist, dict_clusters[C1], dict_clusters[C2])
            )
    return np.min(list_dist)


def greatest_dist_intra_clusters(dict_dist, dict_clusters):
    clusters = list(dict_clusters.keys())
    list_dist = []
    for i in range(len(clusters)):
        C = clusters[i]
        list_dist.append(dist_intra_in_cluster(dict_dist, dict_clusters[C]))
    return np.max(list_dist)


def Dunn(dict_dist, dict_clusters):
    """Computes the Dunn index of a clustering."""
    dmin = smallest_dist_between_clusters(dict_dist, dict_clusters)
    dmax = greatest_dist_intra_clusters(dict_dist, dict_clusters)
    return dmin / dmax


def centroid(points):
    """Computes the centroid of a cluster."""
    return np.mean(points, axis=0)


def intra_cluster_distance(dict_dist, C):
    """Computes the average intra-cluster distance (sigma)."""
    if len(C) == 1:
        return 0
    dist_tmp = []
    for i in range(len(C) - 1):
        for j in range(i + 1, len(C)):
            dist_tmp.append(dict_dist[tuple(sorted((C[i], C[j])))])
    return np.mean(dist_tmp)


def inter_cluster_distance(centroid1, centroid2):
    """Computes the Euclidean distance between two centroids."""
    return np.linalg.norm(np.array(centroid1) - np.array(centroid2))


def Davies_Bouldin(dict_dist, dict_clusters):
    """Computes the Davies-Bouldin index of a clustering."""
    cluster_ids = list(dict_clusters.keys())
    num_clusters = len(cluster_ids)

    centroids = {
        cluster_id: centroid(dict_clusters[cluster_id]) for cluster_id in cluster_ids
    }
    sigmas = {
        cluster_id: intra_cluster_distance(dict_dist, dict_clusters[cluster_id])
        for cluster_id in cluster_ids
    }

    db_index = 0.0
    for i in cluster_ids:
        max_ratio = float("-inf")
        for j in cluster_ids:
            if i != j:
                centroid_dist = inter_cluster_distance(centroids[i], centroids[j])
                ratio = (sigmas[i] + sigmas[j]) / centroid_dist
                if ratio > max_ratio:
                    max_ratio = ratio
        db_index += max_ratio

    return db_index / num_clusters

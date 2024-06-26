import networkx as nx
import numpy as np


def phylogenetic_tree(alphabets_short, df_info, clusters, dict_sim):
    clusters_tmp_script = [alphabets_short[i] for i in clusters]
    time_range = [
        df_info[df_info["Script"] == script]["begin nb"].values[0]
        for script in clusters_tmp_script
    ]
    time_range_sorted = list(np.argsort(time_range))
    clusters_tmp_sorted_by_time = [clusters[i] for i in time_range_sorted]

    labeldict = dict()
    for i in range(len(clusters)):
        labeldict[clusters[i]] = clusters_tmp_script[i]

    G = nx.DiGraph()
    for i in range(len(clusters_tmp_sorted_by_time)):
        cluster = clusters_tmp_sorted_by_time[i]
        clusters_before = clusters_tmp_sorted_by_time[:i]
        if len(clusters_before) == 0:
            G.add_node(labeldict[cluster])
        else:
            S = [
                dict_sim[(tuple(sorted((cluster, cluster_before))))]
                for cluster_before in clusters_before
            ]
            if np.min(S) < 0.25:
                parent = clusters_before[np.argmin(S)]
                G.add_edge(labeldict[parent], labeldict[cluster], weight=np.min(S))
            else:
                G.add_node(labeldict[cluster])

    return G

#!/usr/bin/env python
# coding: utf-8

##################################################################################################
# # Importing

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_moons

# import warnings
# warnings.filterwarnings("ignore")

##################################################################################################
# # Generating Dataset


# generate 2d classification dataset
X, y = make_blobs(n_samples=120, centers=4, n_features=2, cluster_std=1.8, random_state=42)

X1, y1 = make_moons(n_samples=80, noise=0.05, random_state=42)

varied = make_blobs(n_samples=120,
                    cluster_std=[3.5, 3.5, 3.5],
                    random_state=42)[0]
plt.scatter(varied[:, 0], varied[:, 1])
# plt.gcf().gca().add_artist(plt.Circle((-5, 0), 5, color="red", fill=False, linewidth=3, alpha=0.7))
plt.show()


##################################################################################################
# # OPTICS

from algorithms.optics import OPTICS, plot_clust

ClustDist, CoreDist = OPTICS(X, eps=0.5, minPTS=3, plot=True, plot_reach=True)

plot_clust(X, ClustDist, CoreDist, eps=2, eps_db=1)

##################################################################################################
# # DBSCAN

# from algorithms.dbscan import DBSCAN, plot_clust_DB

# ClustDict = DBSCAN(X, eps=2, minPTS=3, plotting=True)

# ClustDict = DBSCAN(X, eps=1.5, minPTS=3, plotting=False)

# plot_clust_DB(X, ClustDict, eps=1.5, noise_circle=True, circle_class="true")

##################################################################################################
# # HDBSCAN

# clusterer = hdbscan.HDBSCAN(min_cluster_size=3, gen_min_span_tree=True)
# clusterer.fit(X)
#
# plt.figure(figsize=(18, 8))
# clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
#                                       edge_alpha=0.6,
#                                       node_size=120,
#                                       edge_linewidth=3,
#                                       )
#
# xmin, xmax, ymin, ymax = plt.axis()
# xwidth = xmax - xmin
# ywidth = ymax - ymin
#
# xw1 = xwidth * 0.015
# yw1 = ywidth * 0
#
# xw2 = xwidth * 0.01
# yw2 = ywidth * 0
#
# for i, txt in enumerate([i for i in range(len(X))]):
#     if len(str(txt)) == 2:
#         plt.annotate(txt, (X[:, 0][i] + xw1, X[:, 1][i] - yw1), fontsize=12, size=12)
#     else:
#         plt.annotate(txt, (X[:, 0][i] + xw2, X[:, 1][i] - yw2), fontsize=12, size=12)
#
# plt.show()
#
#
# # distances in detail
# dist_df = clusterer.minimum_spanning_tree_.to_pandas()
# dist_df = dist_df.sort_values("distance", ascending=False)
#
# # the scale is in log2, it proceeds from the top to the bottom
# plt.figure(figsize=(18, 8))
# clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
# plt.show()
#
# plt.figure(figsize=(18, 8))
# clusterer.condensed_tree_.plot()
# clust_data = clusterer.condensed_tree_.get_plot_data()["cluster_bounds"]
#
# xmin, xmax, ymin, ymax = plt.axis()
# xwidth = xmax - xmin
# ywidth = ymax - ymin
#
# for name in list(clust_data.keys()):
#     data = clust_data[name]
#     x = (data[0] + data[1]) / 2 - xwidth * 0.01
#     y = (data[3]) - ywidth * 0.04
#     plt.annotate("{0}".format(name), (x, y), fontsize=15, size=15, color="black")
#
# plt.show()
#
# plt.figure(figsize=(18, 8))
# clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
# clust_data = clusterer.condensed_tree_.get_plot_data()["cluster_bounds"]
#
# xmin, xmax, ymin, ymax = plt.axis()
# xwidth = xmax - xmin
# ywidth = ymax - ymin
#
# for name in list(clust_data.keys()):
#     data = clust_data[name]
#     x = (data[0] + data[1]) / 2 - xwidth * 0.01
#     y = (data[3]) - ywidth * 0.04
#     plt.annotate("{0}".format(name), (x, y), fontsize=15, size=15)
# plt.show()
#
# plt.figure(figsize=(18, 8))
# palette = sns.color_palette()
# cluster_colors = [sns.desaturate(palette[col], sat)
#                   if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
#                   zip(clusterer.labels_, clusterer.probabilities_)]
# plt.scatter(X.T[0], X.T[1], c=cluster_colors, s=400, edgecolor="black")
#
# xmin, xmax, ymin, ymax = plt.axis()
# xwidth = xmax - xmin
# ywidth = ymax - ymin
#
# xw1 = xwidth * 0.008
# yw1 = ywidth * 0.008
#
# xw2 = xwidth * 0.005
# yw2 = ywidth * 0.008
#
# for i, txt in enumerate([i for i in range(len(X))]):
#     if len(str(txt)) == 2:
#         plt.annotate(txt, (X[:, 0][i] - xw1, X[:, 1][i] - yw1), fontsize=12, size=12)
#     else:
#         plt.annotate(txt, (X[:, 0][i] - xw2, X[:, 1][i] - yw2), fontsize=12, size=12)
#
# plt.show()

##################################################################################################
# # HIERARCHICAL AGGLOMERATIVE CLUSTERING

# from algorithms.agglomerative import agg_clust, agg_clust_mod

# agg_clust(X, "single")
# agg_clust_mod(X, "single")

# agg_clust(X, "complete")
# agg_clust_mod(X, "complete")

# agg_clust(X, "average")
# agg_clust_mod(X, "average")

# agg_clust(X, "ward")
# agg_clust_mod(X, "ward")


##################################################################################################
# # CURE

# from algorithms.cure import cure, plot_results_cure, cure_sample_part, Chernoff_Bounds, demo_parameters

# clusters, rep, mat_a= cure(X, 3, c=4, alpha=0.6)

# clusters, rep, mat_a = cure(varied, 3, c=4, alpha=0.5, plotting=True)

# clusters, rep, mat_a = cure(varied,3, c=7, alpha=0.85, plotting=False)

# plot_results_cure(clusters)

# cure_sample_part(X,c=5, alpha=0.1, k=3)

# Chernoff_Bounds(u_min=500, f=0.5, N=20000, k=2, d=0.05) 

# demo_parameters()

##################################################################################################
# # BIRCH

# from algorithms.birch.birch import birch, plot_birch_leaves, plot_tree_fin

# from algorithms.birch.cftree import measurement_type
# from pyclustering.cluster import cluster_visualizer


# birch_instance = birch(X.tolist(), 3, initial_diameter=4, max_node_entries=5,
#                       type_measurement=measurement_type.CENTROID_EUCLIDEAN_DISTANCE)

# birch_instance.process(plotting=True)

# plot_tree_fin(birch_instance.return_tree(), info=True)

# plot_birch_leaves(birch_instance.return_tree(), X)

# birch_instance.return_tree().show_feature_destibution()

# clusters = birch_instance.get_clusters()
# visualizer = cluster_visualizer()
# visualizer.append_clusters(clusters, X.tolist())
# visualizer.show()

##################################################################################################
# # PAM

# from algorithms.pam import KMedoids
# z = KMedoids(n_cluster=3, tol=0.01)
# z.fit(X.tolist())

##################################################################################################
# # CLARA

# beware that input data dimension must be at least 40+2*n_clusters, otherwise it is useless

# from algorithms.clara import ClaraClustering
# Clara = ClaraClustering()
# final_result = Clara.clara(pd.DataFrame(X), 3, 'fast_euclidean')

##################################################################################################
# # CLARANS

# from algorithms.clarans import clarans, plot_tree_clarans
# plot_tree_clarans(pd.DataFrame(X[:5]),3)

# z = clarans(X, 3, 5, 6).process(plotting=True)

# z.get_clusters()

##################################################################################################
# # CHAMELEON & CHAMELEON2

# ## CHAMELEON 

# from algorithms.chameleon.visualization import plot2d_data
# from algorithms.chameleon.chameleon import cluster

# df = pd.DataFrame(X)

# res, h = cluster(df, k=1, knn=15, m=10, alpha=2, plot=True)

# print("FINAL")
# plot2d_data(res)

# ## CHAMELEON2

# from algorithms.chameleon.visualization import plot2d_data
# from algorithms.chameleon.chameleon2 import cluster2

# df = pd.DataFrame(varied)
# df = pd.DataFrame(X)

# according to the paper, standard number of partitions
# num_part = int(round(len(df) / max(5, round(len(df) / 100))))
# print(num_part)

# res, h = cluster2(df, k=1, knn=None, m=15, alpha=2, beta=1, m_fact=1000, plot=True, auto_extract=True)

# print("\n")
# print("FINAL")
# plot2d_data(res)

##################################################################################################
# # DENCLUE

# from algorithms.denclue.denclue import DENCLUE, plot_3d_or_contour, plot_3d_both, plot_grid_rect, plot_infl

# plot_3d_or_contour(varied, s=0.75, three=True, scatter=True, prec=10)

# plot_3d_or_contour(varied, s=2, three=False, scatter=True, prec=10)

# plot_infl(varied, s=0.75, xi=3)

# plot_3d_both(data=varied, s=2, xi=3, prec=10)

# plot_grid_rect(varied, s=2, cube_kind="highly_populated")

# lab = DENCLUE(data=varied, s=1.2, xi=2, xi_c=3, tol=2, plotting=True)

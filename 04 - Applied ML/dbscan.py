from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.05, min_samples=10)
dbscan.fit(x_new)
core_samples_mask = np.zeros_like(y_new, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True
labels = dbscan.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

y_new_list = list(y_new)
labels = list(labels)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(y_new_list, labels))
print("Completeness: %0.3f" % metrics.completeness_score(y_new_list, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(y_new_list, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(y_new_list, labels))
print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(y_new_list, labels))
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(x_new, labels))

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = 'k'
    class_member_mask = (labels == k)
    
    xy = x_new[class_member_mask & core_samples_mask]
    print(type(xy))
    plt.plot(xy.ix[:, 0], xy.ix[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=8)

    xy = x_new[class_member_mask & ~core_samples_mask]
    plt.plot(xy.ix[:, 0], xy.ix[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=4)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
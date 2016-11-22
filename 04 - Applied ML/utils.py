import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.cm as cm

def fill_nan_values(col):
	"""
	Fills NaN values of columns with the column median

	Parameters
	----------
	col: column of data
	"""
	median_col = np.median(col.dropna())
	return col.fillna(value=median_col)

def choose_best_model(scores, parameters, max_accuracy, tolerance):
	"""
	chooses the simplest model that has an accuracy no worse than a certain threshold compared to the best obtained model

	Parameters:
		scores: list of scores
		parameters: list of parameters used to train the classifier (e.g. list with possible number of trees, list of possible tree depths)
		max_accuracy: best accuracy achieved for some value in the parameters list
		tolerance: threshold value

	Returns the parameter value that achieved a tolerated accuracy 
	"""
	for score in scores:
		if max_accuracy - score <= tolerance:
			s = score
			break
	return parameters[scores.index(s)]

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
						n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
	"""
	Generate a simple plot of the test and training learning curve.

	Parameters
	----------
	estimator : object type that implements the "fit" and "predict" methods
		An object of that type which is cloned for each validation.

	title : string
		Title for the chart.

	X : array-like, shape (n_samples, n_features)
		Training vector, where n_samples is the number of samples and
		n_features is the number of features.

	y : array-like, shape (n_samples) or (n_samples, n_features), optional
		Target relative to X for classification or regression;
		None for unsupervised learning.

	ylim : tuple, shape (ymin, ymax), optional
		Defines minimum and maximum yvalues plotted.

	cv : int, cross-validation generator or an iterable, optional
		Determines the cross-validation splitting strategy.
		Possible inputs for cv are:
		  - None, to use the default 3-fold cross-validation,
		  - integer, to specify the number of folds.
		  - An object to be used as a cross-validation generator.
		  - An iterable yielding train/test splits.

		For integer/None inputs, if ``y`` is binary or multiclass,
		:class:`StratifiedKFold` used. If the estimator is not a classifier
		or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

		Refer :ref:`User Guide <cross_validation>` for the various
		cross-validators that can be used here.

	n_jobs : integer, optional
		Number of jobs to run in parallel (default 1).
	"""
	plt.figure()
	plt.title(title)
	if ylim is not None:
		plt.ylim(*ylim)
	plt.xlabel("Training examples")
	plt.ylabel("Score")
	train_sizes, train_scores, test_scores = learning_curve(
		estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	plt.grid()

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
					 train_scores_mean + train_scores_std, alpha=0.1,
					 color="r")
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
					 test_scores_mean + test_scores_std, alpha=0.1, color="g")
	plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
			 label="Training score")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
			 label="Cross-validation score")

	plt.legend(loc="best")
	return plt

def plot_feature_importances(X, forest):
	"""
	Generate a simple plot of the feature importance.

	Parameters
	----------
	X: features
	
	forest : a trained random forest classifier
	"""
	importances = forest.feature_importances_
	std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
	indices = np.argsort(importances)[::-1]

	# Print the feature ranking
	print("Feature ranking:")

	for f in range(X.shape[1]):
		print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

	# Plot the feature importances of the forest
	plt.figure()
	plt.title("Feature importances")
	plt.bar(range(X.shape[1]), importances[indices],
		   color="r", yerr=std[indices], align="center")
	plt.xticks(range(X.shape[1]), indices)
	plt.xlim([-1, X.shape[1]])

def visualize_pca(data):
    """
    This function is used to visualize data using PCA.
    
    Parameters
        data: dataframe that contains the 2 principal components and the label for each data point.
    """
    plt.figure()
    colors = ['darkorange', 'navy',]
    lw = 2
    target_names = ['white', 'black']
    for color, label, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(data.loc[data.rater1 == label, 'comp1'], data.loc[data.rater1 == label, 'comp2'], color=color, 
                    alpha=.8, lw=lw, label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA on dataset')
    plt.xlabel('First principal compoment')
    plt.ylabel('Second principal compoment')
    plt.grid()
    plt.show()

def plot_hist_of_features(X):
    """
    Plot the histogram for each feature
    
    Parameters:
        X: matrix with features
    """
    features = X.columns
    fig, axes = plt.subplots(7, 2, figsize=(15, 15))
    axes = axes.ravel()
    feature_id = 0
    for idx, ax in enumerate(axes):
        ax.hist(X[features[feature_id]], bins=50)
        title = 'Distribution of ' + features[feature_id]
        ax.set_title(title)
        ax.set_xlabel(features[feature_id])
        ax.set_ylabel('Frequency')
        feature_id += 1
    plt.tight_layout()

def detect_outliers(data, y):
    """
    Detect and remove outliers from a dataset
    
    Parameters:
        data: X matrix with features
        y: labels for each player
    
    Returns:
        a new X and y matrix without rows that contained outliers
    """
    # define the 0.05 and 0.95 quantiles as thresholds
    low = .05
    high = .95
    quant_data = data.quantile([low, high])
    # keep data that are in the interval [0.05, 0.95]
    data = data.apply(lambda col: col[(col > quant_data.loc[low, col.name]) & (col < quant_data.loc[high, col.name])], axis=0)
    # concatenate data with y
    data = pd.concat([y, data], axis=1)
    # keep only rows without NaN values
    data.dropna(inplace=True)
    return data['rater1'], data.loc[:, data.columns != 'rater1']

def find_silhouette(X, features):
    """
    calculates silhouette score for the clustering algorithm
    
    Parameters:
        X: matrix of features
        features: set of features used in the X matrix
        
    Returns:
        silhouette score
    """
    x = pd.DataFrame(X[features])
    # initialize kmeans
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(x)
    # compute score
    silhouette_score = metrics.silhouette_score(x, kmeans.labels_, metric='sqeuclidean')
    # print(features, silhouette_score)
    return silhouette_score

def forward_feature_selection(X):
    """
    Performs forward feature selection. It starts by clustering the data using one
    feature and then adds one by one the most significant featutes.
    The significance level is defined by the silhouette score
    
    Parameters:
        X: X matrix with features
        
    Returns:
        the dictionaty of scores and feature combination
    """

    features = list(X.columns)
    # initially all features are avalable
    available_features = list(X.columns)
    # initially no feature is considered to be best
    best_features = []
    # dictionary: key = silhouette score | value = feature combination
    results = {}
    for i in range(1, len(features) + 1):
        scores = []
        # iterate through all available features
        for feature in available_features:
            # add one feature to the best_features each time
            f = best_features + [feature]
            # calculate score for the f features set
            silhouette = find_silhouette(X, f)
            scores.append(silhouette)
        # keep the max score
        max_silhouette = np.max(scores)
        # add the feature that contributes the most in the score to the best features so far
        best_features_combination = best_features + [available_features[scores.index(max_silhouette)]]
        # remove this feature from the available_features as well
        available_features.remove(available_features[scores.index(max_silhouette)])
        # add the feature that contributes the most in the score to the best features so far
        best_features.append(best_features_combination[-1])
        # enter best score in the dictionary
        results[max_silhouette] = best_features_combination
        # print log message
        num_of_features = str(len(best_features_combination))
        print('---------------------------------------------------------')
        print('Clustering with ' + num_of_features + ' feature(s)')
        print('Best feature combination:\n', '\t', ', '.join(best_features_combination))
        print('Silhouette score: ' ,max_silhouette, '\n')
    return results

def backward_feature_selection(X):
    """
    Performs backward feature selection. It starts by clustering the data using all
    available features and then removes one by one the least significant featutes.
    The significance level is defined by the silhouette score
    
    Parameters:
        X: X matrix with features
    
    Returns:
        the dictionaty of scores and feature combination
    """
    
    features = list(X.columns)
    # initially all features are avalable
    available_features = list(X.columns)
    # initially all features are considered to be best
    best_features = list(features)
    # dictionary: key = silhouette score | value = feature combination
    results = {}
    for i in range(1, len(features)):
        scores = []
        # iterate through all available features
        for feature in available_features:
            f = list(best_features)
            # from the best_features remove one feature each time
            f.remove(feature)
            # calculate score for the f features set
            silhouette = find_silhouette(X, f)
            scores.append(silhouette)
        # keep the max score
        max_silhouette = np.max(scores)
        best_features_combination = list(best_features)
        # from the best features so far, remove the one that contributes the least in the score
        best_features_combination.remove(available_features[scores.index(max_silhouette)])
        # remove this feature from the best_features
        best_features.remove(available_features[scores.index(max_silhouette)])
        # remove this feature from the available_features as well
        available_features.remove(available_features[scores.index(max_silhouette)])
        # enter best score in the dictionary
        results[max_silhouette] = best_features_combination
        # print log message
        num_of_features = str(len(best_features_combination))
        print('---------------------------------------------------------')
        print('Clustering with ' + num_of_features + ' feature(s)')
        print('Best feature combination:\n', '\t', ', '.join(best_features_combination))
        print('Silhouette score: ' ,max_silhouette, '\n')
    return results

def visualize_silhouette_score(ffs):
    """
    visualizes the silhouette score with respect to the number of features
    
    Parameters:
        ffs: dictionary with key = silhouette score and value = feature combination
    """
    x_data = []
    y_data = []
    for key, value in ffs.items():
        y_data.append(key)
        x_data.append(len(value))
    xy = list(zip(x_data, y_data))
    xy = sorted(xy, key=lambda tup: tup[0])
    x_data = [item[0] for item in xy]
    y_data = [item[1] for item in xy]

    p3 = np.poly1d(np.polyfit(x_data, y_data, 3))
    plt.plot(x_data, y_data, '.', x_data, p3(x_data), '--')
    plt.title('Silhouette score vs. number of features')
    plt.xlabel('Number of features')
    plt.ylabel('Silhouette')
    plt.grid()
    plt.show()

def visualize_clusters(kmeans, X, features, n_clusters):
    """
    Plots the 2 clusters of the kmeans algorithm
    
    Parameters:
        kmeans: trained kmeans estimator
        X: matrix with features (n x 2 matrix with the two most important features)
        n_clusters: number of clusters
    """
    colors = ['#4EACC5', '#FF9C34']
    plt.figure()
    plt.hold(True)
    for k, col in zip(range(n_clusters), colors):
        # keep members of each cluster
        my_members = kmeans.labels_ == k
        # keep centers of clusters
        cluster_center = kmeans.cluster_centers_[k]
        # plot first and second feature for each cluster
        plt.plot(X.ix[my_members, 0], X.ix[my_members, 1], 'w',
                markerfacecolor=col, marker='.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=6)
    plt.title('KMeans')
    plt.xlabel(features[0])
    plt.ylabel(features[1])    
    plt.grid(True)
    plt.show()

def silhouette_analysis(X):

    n_clusters = 2
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = metrics.silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = metrics.silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X.ix[:, 0], X.ix[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors)

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1],
                marker='o', c="white", alpha=1, s=200)

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()

def plot_gmm(gmix, X, features):
    """
    visualizes clusters produced using GMM
    
    Parameters:
        gmix: GMM estimator
        samples: matrix with X features
    """
    colors = ['r' if i==0 else 'g' for i in gmix.predict(X)]
    ax = plt.gca()
    ax.scatter(X.ix[:,0], X.ix[:,1], c=colors, alpha=0.8)
    plt.title('Gaussian Mixture Model')
    plt.xlabel(features[0])
    plt.ylabel(features[1]) 
    plt.show()
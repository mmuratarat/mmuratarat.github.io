---
layout: post
title: "Implementing K-means Clustering from Scratch - in Python"
author: "MMA"
comments: true
---

# K-means Clustering

K-means algorithm is is one of the simplest and popular unsupervised machine learning algorithms, that solve the well-known clustering problem, with no pre-determined labels defined, meaning that we don’t have any target variable as in the case of supervised learning. 

K-means simply partitions the given dataset into various clusters (groups).

K refers to the total number of clusters to be defined in the entire dataset.There is a centroid chosen for a given cluster type which is used to calculate the distance of a given data point. The distance essentially represents the similarity of features of a data point to a cluster type.

You’ll define a target number K, which refers to the number of centroids you need in the dataset. A centroid is the imaginary or real location representing the center of the cluster. These centroids shoud be placed in a cunning way because of different location causes different result. So, the better choice is to place them as much as possible far away from each other. 

In other words, the K-means algorithm identifies K number of centroids, and then allocates every data point to the nearest cluster, while keeping the centroids as small as possible. The 'means' in the K-means refers to averaging of the data; that is, finding the centroid.

Every data point is allocated to each of the clusters through reducing the in-cluster sum of squares. Once the algorithm has been run and the groups are defined, any new data can be easily assigned to the correct group.

In K-means, each cluster is described by a single mean, or centroid (hard clustering), so as not to confuse this model with an actual probabilistic model. **There is no underlying probability model in K-means**. The goal is to group data into K clusters. K-means (and some other clustering methods) have hard boundaries, meaning a data point either belongs to that cluster or it does not. On the other hand, clustering methods such as Gaussian Mixture Models (GMM) have soft boundaries (soft clustering), where data points can belong to multiple cluster at the same time but with different degrees of belief. e.g. a data point can have a $60\%$ of belonging to cluster $1$, $40\%$ of belonging to cluster $2$. Additionally, in probabilistic clustering, clusters can overlap (K-means doesn’t allow this).

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202019-10-05%20at%2016.22.34.png?raw=true)

An important observation for k-means is that the cluster models must be circular (or spherical in high dimensions, i.i.d. Gaussian). In other words, K-means requires that each blob be a fixed size and completely symmetrical. K-means has no built-in way of accounting for oblong or elliptical clusters. When clusters are non-circular, trying to fit circular clusters would be a poor fit. This results in a mixing of cluster assignments where the resulting circles overlap.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/circular_clusters.png?raw=true)

These two disadvantages of K-means—its lack of flexibility in cluster shape and lack of probabilistic cluster assignment—mean that for many datasets (especially low-dimensional datasets) it may not perform as well as you might hope. K-means is also very sensitive to outliers and noise in the dataset.

# When to use?
This is a versatile algorithm that can be used for any type of grouping. Some examples of use cases are:
1. Image Segmentation
2. Clustering Gene Segementation Data
3. News Article Clustering
4. Clustering Languages
5. Species Clustering
6. Anomaly Detection

# Algorithm

The Κ-means clustering algorithm uses iterative refinement to produce a final result. The algorithm inputs are the number of clusters Κ and the data set. The data set is a collection of features for each data point. 

**Step 1**

The algorithms starts with initial estimates for the Κ centroids, which can either be randomly generated or randomly selected from the data set. Random initialization is not an efficient way to start with, as sometimes it leads to increased numbers of required clustering iterations to reach convergence, a greater overall runtime, and a less-efficient algorithm overall. So there are many techniques to solve this problem like K-means++ etc. 

We randomly pick K cluster centers(centroids). Let’s assume these are $c_1, c_2, ..., c_K$, and we can say that;

$$C = {c_1, c_2,..., c_K}$$

where $C$ is the set of all centroids.

**Step 2**

Each centroid defines one of the clusters. In this step, each data point is assigned to its nearest centroid, based on the squared Euclidean distance. More formally, if $c_{i}$ is the collection of centroids in set $C$, then each data point $x$ is assigned to a cluster based on

$$\underset{c_i \in C}{\arg\min} \; dist(c_i,x)^2$$

where $dist( \cdot )$ is the standard (L2) Euclidean distance. Let the set of data point assignments for each ith cluster centroid be $S_{i}$. Note that the distance function in the cluster assignment step can be chosen specifically for your
problem, and is arbitrary.

**Step 3**

In this step, the centroids are recomputed. This is done by taking the mean of _all data points_ assigned to that centroid's 
cluster.

$$c_i=\frac{1}{\lvert S_i \rvert}\sum_{x_i \in S_i} x_i$$

where $S_{i}$ is the set of all points assigned to the $i$th cluster.

**Step 4**

The algorithm iterates between steps one and two until a stopping criteria is met (i.e., no data points change clusters, the sum of the distances is minimized, or some maximum number of iterations is reached).

The best number of clusters K leading to the greatest separation (distance) is not known as a priori and must be computed from the data. The objective of K-Means clustering is to minimize total intra-cluster variance, or, the squared error function: 

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Clustering_kmeans_c.png?raw=true)

**NOTE**: Unfortunately, although the algorithm is guaranteed to converge, it may not converge to the right solution (i.e., it may converge to a local optimum, not necessarily the best possible outcome). This highly depends on the centroid initialization. As a result, the computation is often done several times, with different initializations of the centroids. One method to help address this issue is the K-means++ initialization scheme, which has been implemented in scikit-learn (use the `init='k-means++'` parameter). This initializes the centroids to be (generally) distant from each other, leading to probably better results than random initialization. One idea for initializing K-means is to use a farthest-first traversal on the data set, to pick K points that are far away from each other. However, this is too sensitive to outliers. But, K-means++ procedure picks the K centers one at a time, but instead of always choosing the point farthest from those picked so far, choose each point at random, with probability proportional to its squared distance from the centers chosen already. 

**NOTE**: The computational complexity of the algorithm is generally linear with regards to the number of instances, the number of clusters and the number of dimensions. However, this is only true when the data has a clustering structure. If it does not, then in the worst case scenario the complexity can increase exponentially with the number of instances. In practice, however, this rarely happens, and K-Means is generally one of the fastest clustering algorithms.

# Choosing the Value of K
We often know the value of K. In that case we use the value of K. In general, there is no method for determining exact value of K, but an accurate estimate can be obtained using the Elbow Method. We run the algorithm for different values of K (say K = 1 to 10) and plot the K values against WCSSE (Within Cluster Sum of Squared Errors). WCSS is also called "inertia". Then, select the value of K that causes sudden drop in the sum of squared distances, i.e., for the elbow point as shown in the figure.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/elbow_method_kmeans.png?raw=true)

A number of other techniques exist for validating K, including cross-validation, information criteria, the information theoretic jump method, the silhouette method (we want to have high silhouette coefficient for the number of clusters we want to use), and the G-means algorithm. In addition, monitoring the distribution of data points across groups provides insight into how the algorithm is splitting the data for each K. Some researchers also use Hierarchical clustering first to create dendrograms and identify the distinct groups from there.

## Constraints of the algorithm

Only numerical data can be used. Generally K-means works best for 2 dimensional numerical data. Visualization is possible in 2D or 3D data. But in reality there are always multiple features to be considered at a time. However, we must be careful about curse of dimensionality. any more than few tens of dimensions mean that distance interpretation isn’t obvious and must be guarded against. Appropriate dimensionality reduction techniques and distance measure must be used.

K-Means clustering is prone to initial seeding i.e. random initialization of centroids which is required to kick-off iterative clustering process. Bad initialization may end up getting bad clusters.
 
The standard K-means algorithm isn't directly applicable to categorical data, for various reasons. The sample space for categorical data is discrete, and doesn't have a natural origin. A Euclidean distance function on such a space is not really meaningful. However, The clustering algorithm is free to choose any distance metric / similarity score. Euclidean is the most popular. But any other metric can be used that scales according to the data distribution in each dimension/attribute, for example the Mahalanobis metric.

The use of Euclidean distance as the measure of dissimilarity can also make the determination of the cluster means non-robust to outliers and noise in the data.

Categorical data (i.e., category labels such as gender, country, browser type) needs to be encoded (e.g., one-hot encoding for nominal categorical variable or label encoding for ordinal categorical variable) or separated in a way that can still work with the algorithm, which is still not perfectly right. There's a variation of K-means known as K-modes, introduced in [this paper](http://www.cs.ust.hk/~qyang/Teaching/537/Papers/huang98extensions.pdf) by Zhexue Huang, which is suitable for categorical data. 

K-Means does not behave very well when the clusters have varying sizes, different densities, or non-spherical shapes. In that case, one can use Mixture models using EM algorithm or Fuzzy K-means (every object belongs to every cluster with a membershio weight that is between 0 (absolutely does not belong) and 1 (absolutely belongs)). which both allow soft assignments. As a matter of fact, K-means is special variant of the EM algorithm with the assumption that the clusters are spherical. EM algorithm also starts with random initializations, it is an iterative algorithm, it has strong assumptions that the data points must fulfill, it is sensitive to outliers, it requires prior knowledge of the number of desired clusters. The results produced by EM are also non-reproducible.

The above paragraph shows the drawbacks of this algorithm. K-means assumes the variance of the distribution of each attribute (variable) is spherical; all variables have the same variance; the prior probability for all K clusters is the same, i.e., each cluster has roughly equal number of observations. If any one of these 3 assumptions are violated, then K-means will fail. [This Stackoverflow answer](https://stats.stackexchange.com/a/249288/16534) explains perfectly!

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-05-20%20at%2019.08.40.png?raw=true)

It is important to scale the input features before you run K-Means, or else the clusters may be very stretched, and K-Means will perform poorly. Scaling the features does not guarantee that all the clusters will be nice and spherical, but it generally improves things.

K-Means clustering just cannot deal with missing values. Any observation even with one missing dimension must be specially handled. If there are only few observations with missing values then these observations can be excluded from clustering. However, this must have equivalent rule during scoring about how to deal with missing values. Since in practice one cannot just refuse to exclude missing observations from segmentation, often better practice is to impute missing observations. There are various methods available for missing value imputation but care must be taken to ensure that missing imputation doesn’t distort distance calculation implicit in k-Means algorithm. For example, replacing missing age with -1 or missing income with 999999 can be misleading!

Clustering analysis is not negatively affected by heteroscedasticity but the results are negatively impacted by multicollinearity of features/ variables used in clustering as the correlated feature/ variable will carry extra weight on the distance calculation than desired.

K-Means clustering algorithm might converse on local minima which might also correspond to the global minima in some cases but not always. Therefore, it’s advised to run the K-Means algorithm multiple times before drawing inferences about the clusters. However, note that it’s possible to receive same clustering results from K-means by setting the same seed value for each run. But that is done by simply making the algorithm choose the set of same random number for each run.

# DATA: Iris Flower Dataset 

{% highlight python %}
import numpy as np
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
%matplotlib inline

data = load_iris()
X = data['data']
y = data['target']

# Number of training data
n = X.shape[0]
# Number of features in the data
c = X.shape[1]

# Plot the data
colors=['orange', 'blue', 'green']
for i in range(n):
    plt.scatter(X[i, 0], X[i,1], s=7, color = colors[int(y[i])])
{% endhighlight %}

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/iris_clusters.png?raw=trueg)

# K-means in Sci-kit Learn  

{% highlight python %}
Kmean = KMeans(n_clusters=3)
Kmean.fit(X)
centers = Kmean.cluster_centers_
# array([[5.006     , 3.418     , 1.464     , 0.244     ],
#        [5.9016129 , 2.7483871 , 4.39354839, 1.43387097],
#        [6.85      , 3.07368421, 5.74210526, 2.07105263]])

# Plot the data
colors=['orange', 'blue', 'green']
for i in range(n):
    plt.scatter(X[i, 0], X[i,1], s=7, color = colors[int(y[i])])
plt.scatter(centers[:,0], centers[:,1], marker='*', c='g', s=150)

y_pred_clusters = Kmean.labels_
# array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0,
#        0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2, 2, 0, 0, 0, 0,
#        0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2], dtype=int32)
{% endhighlight %}

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/iris_scikitlearn_kmeans_clusters.png?raw=true)

# K-means from Scratch

{% highlight python %}
from copy import deepcopy

# Number of clusters
K = 3
# Number of training data
n = X.shape[0]
# Number of features in the data
c = X.shape[1]

# Generate random centers, here we use sigma and mean to ensure it represent the whole data
mean = np.mean(X, axis = 0)
std = np.std(X, axis = 0)
centers = np.random.randn(K,c)*std + mean

centers_old = np.zeros(centers.shape) # to store old centers
centers_new = deepcopy(centers) # Store new centers

clusters = np.zeros(n)
distances = np.zeros((n,K))

error = np.linalg.norm(centers_new - centers_old)

# When, after an update, the estimate of that center stays the same, exit loop
while error != 0:
    # Measure the distance to every center
    for i in range(K):
        distances[:,i] = np.linalg.norm(X - centers_new[i], axis=1)
    # Assign all training data to closest center
    clusters = np.argmin(distances, axis = 1)
    
    centers_old = deepcopy(centers_new)
    # Calculate mean for every cluster and update the center
    for i in range(K):
        centers_new[i] = np.mean(X[clusters == i], axis=0)
    error = np.linalg.norm(centers_new - centers_old)
centers_new    
# array([[5.006     , 3.418     , 1.464     , 0.244     ],
#        [6.48787879, 2.96212121, 5.34242424, 1.87575758],
#        [5.82352941, 2.69705882, 4.05882353, 1.28823529]])

# Plot the data
colors=['orange', 'blue', 'green']
for i in range(n):
    plt.scatter(X[i, 0], X[i,1], s=7, color = colors[int(y[i])])
plt.scatter(centers_new[:,0], centers_new[:,1], marker='*', c='g', s=150)
{% endhighlight %}

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/iris_scratch_kmeans_clusters.png?raw=true)

## k-median

A clustering algorithm closely related to k-means. The practical difference between the two is as follows:

* In k-means, centroids are determined by minimizing the sum of the squares of the distance between a centroid candidate and each of its examples.
* In k-median, centroids are determined by minimizing the sum of the distance between a centroid candidate and each of its examples.

K-medians owes its use to robustness of the median as a statistic. The mean is a measurement that is highly vulnerable to outliers. Even just one drastic outlier can pull the value of the mean away from the majority of the data set, which can be a high concern when operating on very large data sets. The median, on the other hand, is a statistic incredibly resistant to outliers, for in order to deter the median away from the bulk of the information, it requires at least 50% of the data to be contaminated

K-medians uses the median as the statistic to determine the center of each cluster. 

Note that the definitions of distance are also different:

* k-means relies on the Euclidean distance from the centroid to an example. (In two dimensions, the Euclidean distance means using the Pythagorean theorem to calculate the hypotenuse.) For example, the k-means distance between $(2,2)$ and $(5,-2)$ would be:
  
  $$
\text{Euclidean Distance} = \sqrt{(2-5)^{2} + (2 - -2)^{2}} =5
$$

* k-median relies on the Manhattan distance from the centroid to an example. This distance is the sum of the absolute deltas in each dimension. For example, the k-median distance between $(2,2)$ and $(5,-2)$ would be:

  $$
\text{Manhattan Distance} = \lvert 2-5 \rvert + \lvert 2 - -2 \rvert = 7
$$

Note that k-medians is also very sensitive to the initialization points of its k centers, each center having the tendency to remain roughly in the same cluster in which it is first placed.

## Mini Batch K-Means

The Mini-batch K-Means is a variant of the K-Means algorithm which uses mini-batches to reduce the computation time, while still attempting to optimise the same objective function. Mini-batches are subsets of the input data, randomly sampled in each training iteration. These mini-batches drastically reduce the amount of computation required to converge to a local solution. In contrast to other algorithms that reduce the convergence time of K-means, mini-batch K-means produces results that are generally only slightly worse than the standard algorithm.

The algorithm iterates between two major steps, similar to vanilla K-means. In the first step,  samples are drawn randomly from the dataset, to form a mini-batch. These are then assigned to the nearest centroid. In the second step, the centroids are updated. In contrast to k-means, this is done on a per-sample basis. For each sample in the mini-batch, the assigned centroid is updated by taking the streaming average of the sample and all previous samples assigned to that centroid. This has the effect of decreasing the rate of change for a centroid over time. These steps are performed until convergence or a predetermined number of iterations is reached.

Mini-batch K-Means converges faster than K-Means, but the quality of the results is reduced. In practice this difference in quality can be quite small, as shown in the example and cited reference.

For details, look [here](https://scikit-learn.org/stable/modules/clustering.html#mini-batch-k-means){:target="_blank"}

# REFERENCES

1. [https://datascience.stackexchange.com/questions/22/k-means-clustering-for-mixed-numeric-and-categorical-data](https://datascience.stackexchange.com/questions/22/k-means-clustering-for-mixed-numeric-and-categorical-data){:target="_blank"} 
2. [https://blog.bioturing.com/2018/10/17/k-means-clustering-algorithm-and-example/](https://blog.bioturing.com/2018/10/17/k-means-clustering-algorithm-and-example/){:target="_blank"} 
3. [https://www.datascience.com/blog/k-means-clustering](https://www.datascience.com/blog/k-means-clustering){:target="_blank"}
4. [http://worldcomp-proceedings.com/proc/p2015/CSC2663.pdf](http://worldcomp-proceedings.com/proc/p2015/CSC2663.pdf){:target="_blank"}

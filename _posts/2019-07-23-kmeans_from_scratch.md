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

where $dist( \cdot )$ is the standard (L2) Euclidean distance. Let the set of data point assignments for each ith cluster centroid be $S_{i}$.

**Step 3**

In this step, the centroids are recomputed. This is done by taking the mean of all data points assigned to that centroid's 
cluster.

$$c_i=\frac{1}{\lvert S_i \rvert}\sum_{x_i \in S_i} x_i$$

where $S_{i}$ is the set of all points assigned to the $i$th cluster.

**Step 4**

The algorithm iterates between steps one and two until a stopping criteria is met (i.e., no data points change clusters, the sum of the distances is minimized, or some maximum number of iterations is reached).

The best number of clusters K leading to the greatest separation (distance) is not known as a priori and must be computed from the data. The objective of K-Means clustering is to minimize total intra-cluster variance, or, the squared error function: 

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Clustering_kmeans_c.png?raw=true)

**NOTE**: Unfortunately, although the algorithm is guaranteed to converge, it may not converge to the right solution (i.e., it may converge to a local optimum, not necessarily the best possible outcome): this depends on the centroid initialization. 

**NOTE**: The computational complexity of the algorithm is generally linear with regards to the number of instances, the number of clusters and the number of dimensions. However, this is only true when the data has a clustering structure. If it does not, then in the worst case scenario the complexity can increase exponentially with the number of instances. In practice, however, this rarely happens, and K-Means is generally one of the fastest clustering algorithms.

# Choosing the Value of K
We often know the value of K. In that case we use the value of K. In general, there is no method for determining exact value of K, but an accurate estimate can be obtained using the Elbow Method. We run the algorithm for different values of K (say K = 10 to 1) and plot the K values against SSE (Sum of Squared Errors). And select the value of K that causes sudden drop in the sum of squared distances, i.e., for the elbow point as shown in the figure.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/elbow_method_kmeans.png?raw=true)

A number of other techniques exist for validating K, including cross-validation, information criteria, the information theoretic jump method, the silhouette method, and the G-means algorithm. In addition, monitoring the distribution of data points across groups provides insight into how the algorithm is splitting the data for each K. Some researchers also use Hierarchical clustering first to create dendrograms and identify the distinct groups from there.

## Constraints of the algorithm
Only numerical data can be used. Generally K-means works best for 2 dimensional numerical data. Visualization is possible in 2D or 3D data. But in reality there are always multiple features to be considered at a time. 

The standard K-means algorithm isn't directly applicable to categorical data, for various reasons. The sample space for categorical data is discrete, and doesn't have a natural origin. A Euclidean distance function on such a space isn't really meaningful. However, The clustering algorithm is free to choose any distance metric / similarity score. Euclidean is the most popular. But any other metric can be used that scales according to the data distribution in each dimension/attribute, for example the Mahalanobis metric.

Categorical data (i.e., category labels such as gender, country, browser type) needs to be encoded (e.g., one-hot encoding for nominal categorical variable or label encoding for ordinal categorical variable) or separated in a way that can still work with the algorithm, which is still not perfectly right. 

There's a variation of K-means known as K-modes, introduced in [this paper](http://www.cs.ust.hk/~qyang/Teaching/537/Papers/huang98extensions.pdf) by Zhexue Huang, which is suitable for categorical data. 

K-Means does not behave very well when the clusters have varying sizes, different densities, or non-spherical shapes.

It is important to scale the input features before you run K-Means, or else the clusters may be very stretched, and K-Means will perform poorly. Scaling the features does not guarantee that all the clusters will be nice and spherical, but it generally improves things.

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
        distances[:,i] = np.linalg.norm(X - centers[i], axis=1)
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

# REFERENCES

1. [https://datascience.stackexchange.com/questions/22/k-means-clustering-for-mixed-numeric-and-categorical-data](https://datascience.stackexchange.com/questions/22/k-means-clustering-for-mixed-numeric-and-categorical-data){:target="_blank"} 
2. [https://blog.bioturing.com/2018/10/17/k-means-clustering-algorithm-and-example/](https://blog.bioturing.com/2018/10/17/k-means-clustering-algorithm-and-example/){:target="_blank"} 
3. [https://www.datascience.com/blog/k-means-clustering](https://www.datascience.com/blog/k-means-clustering){:target="_blank"}

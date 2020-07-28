---
layout: post
title:  "Principal Component Analysis"
author: "MMA"
comments: true
---

# CURSE OF DIMENSIONALITY

As the number of features (dimensionality) increases, the data becomes relatively more sparse and often exponentially more samples are needed to make statistically significant predictions. 

Curse of Dimensionality refers to the fact that many problems that do not exist in low-dimensional space arise in high-dimensional space. It makes it very difficult to identify the patterns in the data without having plenty of training data because of sparsity of training data in the high dimensional space.

Imagine going from a $10 \times 10$ grid to a $10 \times 10 \times 10$ grid. We want ONE sample in each '$1 \times 1$ square', then the addition of the third parameter requires us to have 10 times as many samples (1000) as we needed when we had 2 parameters (100).

High-dimensional datasets are at risk of being very sparse: most training instances are likely to be far away from each other. Of course, this also means that a new instance will likely be far away from any training instances, making predictions are much less reliable than in lower dimensions, since they will be based on much larger extrapolations. In short, the more dimensions the training set has, the greater the risk of overfitting it.

In theory, one solution to the curse of dimensionality could be to increase the size of the training set to reach a sufficient density of training instances. Unfortunately, in practice, the number of training instances required to reach a given density grows exponentially with the number of dimensions. 

Linear models with no feature selection or regularization, kNN, Bayesian models are models that are most affected by curse of dimensionality. Models that are less affected by the curse of dimensionaliy are regularized models, random forest, some neural networks, stochastic models (e.g. monte carlo simulations).

# PROJECTION

In most real-world problems, training instances are not spread out uniformly across all dimensions. Many features are almost constant, while others are highly correlated. As a result, all training instances actually lie within (or close to) much lower-dimensional _subspace_ of the high dimensional space. 

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/3d_data.png?raw=true)

Notice that all training instances lie close to a plane: this is a lower-dimensional (2D) subspace of the high-dimensional (3D) space. Now if we project every training instance perpendicularly onto this subspace (as represented by the short lines connecting the instances to the plane), we get the new 2D dataset

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/3d_to_2d_projection.png?raw=true)

We have just reduced the dataset’s dimensionality from 3D to 2D. Note that the axes correspond to new features z1 and z2 (the coordinates of the projections on the plane).

However, projection is not always the best approach to dimensionality reduction. In many cases the subspace may twist and turn, such as in the famous Swiss roll toy data‐set 

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/swiss_roll_data.png?raw=true)

Simply projecting onto a plane (e.g., by dropping x3) would squash different layers of the Swiss roll together, as shown on the left. However, what you really want is to unroll the Swiss roll to obtain the 2D dataset on the right. 

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/projection_and_manifold.png?raw=true)

# PRINCIPAL COMPONENT ANALYSIS

Principal Component Analysis (PCA) is by far the most popular dimensionality reduction algorithm. First it identifies the hyperplane that lies closes the data and projects the data onto it. 

## Preserving Variance

Before you can project the training set onto a lower-dimensional hyperplane, you first need to choose the right hyperplane. For example, a simple 2D dataset is represented on the left along with three different axes (i.e., one-dimensional hyperplanes). On the right is the result of the projection of the dataset onto each of these axes. 

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/projection_from_2d_into_line.png?raw=true)

As one can see easily, the projection onto the solid line preserved the maximum variance, while the project onto the dotted line preserves very little variance and the projection onto the dashed line preserves an intermediate amount of variance. It seems reasonable to select the axis which preserved the maximum amount of variance as it will most likely lose less information than other projections. Another way to justify this choice is that it is the axis that minimizes the mean squared distance between the original dataset and its projection onto that axis. "The maximum variance" and "the minimum error" are reached at the same time. This is rather simple idea behind PCA. 

## Principal Components

PCA identifies the axis that accounts for the largest amount of variance in the training set. In the example above it is the solid line. It also finds a second axis, orthogonal to the first one, that accounts for the largest amount of remaining variance. In this 2D example there is no choice: it is the dotted line.  If it were a higher-dimensional dataset, PCA would also find a third axis, orthogonal to both previous axes, and a fourth, a fifth, and so on—as many axes as the number of dimensions in the dataset. 

The unit vector that defines the $i$th axis is called the $i$th principal component (PC). In the example above, the 1st PC is $c_1$ and the 2nd PC is $c_2$. 

Every principal component will ALWAYS be orthogonal (a.k.a. official math term for perpendicular) to every other principal component. Because the principal components are orthogonal to one another, they are statistically linearly independent of one another. Hence, also solving the problem of multicollinearity.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/3d_data.png?raw=true)

Here, the first two PCs are represented by the orthogonal arrows in the plane, and the third PC would be orthogonal to the plane (pointing up or down).

Eigenvalues are the variances of the principal components. In other words, the first eigenvalue is the variance of the first principal component, the second eigenvalue is the variance of the second principal component and so on. Thus, because of the way the principal components are selected, the first eigenvalue will be the largest, the second the next largest etc. There will be $p$ eigenvalues where $p$ is the number of features in the dataset but some may be zero.

NOTE: The direction of the principal components is not stable: if you perturb the training set slightly and run PCA again, some of the new PCs may point in the opposite direction of the original PCs. However, they will generally still lie on the same axes. In some cases, a pair of PCs may even rotate or swap, but the plane they define will generally remain the same.

### How to find PCs?

There is a standard matrix factorization technique called Singular Value Decomposition (SVD) that can decompose the training set matrix $X$ into the dot product of three matrices $U \cdot \Sigma \cdot V^{T}$, where $V^{T}$ contains all the principal components that we are looking for:

$$
V = \begin{bmatrix} \uparrow & \uparrow & \ldots & \uparrow \\
c_{1} & c_{2} & \ldots &  c_{n} \\
\downarrow & \downarrow & \ldots & \downarrow \\
\end{bmatrix}
$$

The following Python code uses NumPy’s `svd()` function to obtain all the principal components of the training set, then extracts the first two PCs:

{% highlight python %} 
import numpy as np
np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

X_centered = X - X.mean(axis=0)
#(60, 3)

U, s, Vt = np.linalg.svd(X_centered, full_matrices=True)
U.shape, s.shape, Vt.shape
#((60, 60), (3,), (3, 3))
c1 = Vt.T[:, 0]
c2 = Vt.T[:, 1]

Sigma = np.diag(s)

# Reconstruction from SVD
# If A matrix is not a square but rectangular matrix, U @ np.diag(s) @ Vt won't work to reconstruct X_centered, 
#you can use this instead:
m, n = X_centered.shape
X_reconstructed = U[:,:n] @ np.diag(s) @ Vt[:m,:]

np.allclose(X_centered, X_reconstructed)
#True

#or you may use 'full_matrices=False' option in the SVD function;

# import numpy as np
# U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
# U.shape, s.shape, Vt.shape
# #((60, 3), (3,), (3, 3))
# X_reconstructed = U @ np.diag(s) @ Vt

# np.allclose(X_centered, X_reconstructed)
# #True
{% endhighlight %}

Note that PCA assumes that the dataset is centered around the origin. As we will see, Scikit-Learn’s PCA classes take care of centering the data for you. However, if you implement PCA yourself (as in the preceding example), or if you use other libraries, don’t forget to center the data first.

### Projecting Down to d Dimensions

Once you have identified all the principal components, you can reduce the dimensionality of the dataset down to d dimensions by projecting it onto the hyperplane defined by the first d principal components. Selecting this hyperplane ensures that the projection will preserve as much variance as possible. To project the training set onto the hyperplane, you can simply compute the dot product of the training set matrix $X$ by the matrix $W_d$, defined as the matrix containing the first $d$ principal components (i.e., the matrix composed of the first d columns of $V^{T}$):

$$
X_{d-projected} = X \cdot W_{d}
$$

The following Python code projects the training set onto the plane defined by the first two principal components:

{% highlight python %} 
W2 = Vt.T[:, :2] #because we have 2 principal components
X2D = X_centered @ W2
#(20, 2)

X2D_using_svd = X2D
{% endhighlight %}

There you have it! You now know how to reduce the dimensionality of any dataset down to any number of dimensions, while preserving as much variance as possible.

# PCA using Scikit-Learn

Scikit-Learn’s PCA class implements PCA using SVD decomposition just like we did before. The following code applies PCA to reduce the dimensionality of the dataset down to two dimensions (note that it automatically takes care of centering the data):

{% highlight python %} 
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
X2D = pca.fit_transform(X)

X2D[:5]
# array([[ 1.26203346,  0.42067648],
#        [-0.08001485, -0.35272239],
#        [ 1.17545763,  0.36085729],
#        [ 0.89305601, -0.30862856],
#        [ 0.73016287, -0.25404049]])

X2D_using_svd[:5]
# array([[-1.26203346, -0.42067648],
#        [ 0.08001485,  0.35272239],
#        [-1.17545763, -0.36085729],
#        [-0.89305601,  0.30862856],
#        [-0.73016287,  0.25404049]])
{% endhighlight %}

Notice that running PCA multiple times on slightly different datasets may result in different results. In general the only difference is that some axes may be flipped. In this example, PCA using Scikit-Learn gives the same projection as the one given by the SVD approach, except both axes are flipped:

{% highlight python %} 
np.allclose(X2D, -X2D_using_svd)
#True
{% endhighlight %}

Recover the 3D points projected on the plane (PCA 2D subspace).

Of course, there was some loss of information during the projection step, so the recovered 3D points are not exactly equal to the original 3D points:

{% highlight python %} 
np.allclose(X3D_inv, X)
#False
{% endhighlight %}

We can compute the reconstruction error:

{% highlight python %} 
np.mean(np.sum(np.square(X3D_inv - X), axis=1))
#0.010170337792848549
{% endhighlight %}

The inverse transform in the SVD approach looks like this:

{% highlight python %} 
X3D_inv_using_svd = X2D_using_svd.dot(Vt[:2, :])
{% endhighlight %}

The reconstructions from both methods are not identical because Scikit-Learn's PCA class automatically takes care of reversing the mean centering, but if we subtract the mean, we get the same reconstruction:

{% highlight python %} 
np.allclose(X3D_inv_using_svd, X3D_inv - pca.mean_)
#True
{% endhighlight %}

The PCA object gives access to the principal components that it computed:

{% highlight python %} 
pca.components_
# array([[-0.93636116, -0.29854881, -0.18465208],
#        [ 0.34027485, -0.90119108, -0.2684542 ]])
{% endhighlight %}

Compare to the first two principal components computed using the SVD method:

{% highlight python %} 
Vt[:2]
# array([[ 0.93636116,  0.29854881,  0.18465208],
#        [-0.34027485,  0.90119108,  0.2684542 ]])
{% endhighlight %}

Notice how the axes are flipped.

Now let's look at the explained variance ratio:

{% highlight python %} 
pca.explained_variance_ratio_
#array([0.84248607, 0.14631839])
{% endhighlight %}

The first dimension explains 84.2% of the variance, while the second explains 14.6%. (total 98.9% percept)

By projecting down to 2D, we lost about 1.1% of the variance:

{% highlight python %} 
1 - pca.explained_variance_ratio_.sum()
#0.011195535570688975
{% endhighlight %}

Here is how to compute the explained variance ratio using the SVD approach (recall that s is the diagonal of the matrix Sigma). Taking square of those values is equal to eigenvalues of $X^{T}X$ (Gram matrix):

{% highlight python %} 
variances = np.square(s) / np.square(s).sum()
# array([0.84248607, 0.14631839, 0.01119554])
{% endhighlight %}

# Choosing the Right Number of Dimensions

Instead of arbitrarily choosing the number of dimensions to reduce down to, it is generally preferable to choose the number of dimensions that add up to a sufficiently large portion of the variance (e.g., 95%). You can plot the explained variance as a function of the number of dimensions (simply plot cumsum), which is called a scree plot:

{% highlight python %} 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#Number of components cannot be greater than number of features in the data
pca = PCA(n_components = 3)
X2D = pca.fit_transform(X)
# pca.explained_variance_ratio_
plt.plot(np.cumsum(pca.explained_variance_ratio_))
{% endhighlight %}

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/scree_scikit_pca.png?raw=true)

or using SVD approach:

{% highlight python %} 
import matplotlib.pyplot as plt

cumsum = np.cumsum(variances)

plt.plot(cumsum)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
{% endhighlight %}

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/scree_svd_pca.png?raw=true)

# PCA for Compression

{% highlight python %} 
from six.moves import urllib
try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1)
    mnist.target = mnist.target.astype(np.int64)
except ImportError:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')
    
from sklearn.model_selection import train_test_split

X = mnist["data"]
y = mnist["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1

plt.plot(cumsum)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
{% endhighlight %}

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/scree_compression.png?raw=true)

This curve quantifies how much of the total, 784-dimensional variance is contained within the first N components. For example, we see that with the digits the first 154 components contain approximately 95% of the variance.

{% highlight python %} 
d
#154
{% endhighlight %}

PCA can also be used as a filtering approach for noisy data. The idea is this: any components with variance much larger than the effect of the noise should be relatively unaffected by the noise. So if you reconstruct the data using just the largest subset of principal components, you should be preferentially keeping the signal and throwing out the noise.

Obviously after dimensionality reduction, the training set takes up much less space. For example, try applying PCA to the MNIST dataset while preserving 95% of its variance. You should find that each instance will have just over 150 features, instead of the original 784 features. So while most of the variance is preserved, the dataset is now less than 20% of its original size! This is a reasonable compression ratio, and you can see how this can speed up a classification algorithm (such as an SVM classifier) tremendously.

It is also possible to decompress the reduced dataset back to 784 dimensions by applying the inverse transformation of the PCA projection. Of course this won’t give you back the original data, since the projection lost a bit of information (within the 5% variance that was dropped), but it will likely be quite close to the original data. The mean squared distance between the original data and the reconstructed data (compressed and then decompressed) is called the reconstruction error. 

{% highlight python %} 
pca = PCA(n_components = 154)
X_reduced_pca = pca.fit_transform(X_train)
#(52500, 154)
X_recovered = pca.inverse_transform(X_reduced_pca)
#(52500, 784)

def plot_digits(instances, images_per_row=5, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, **options)
    plt.axis("off")
    
plt.figure(figsize=(7, 4))
plt.subplot(121)
plot_digits(X_train[::2100])
plt.title("Original", fontsize=16)
plt.subplot(122)
plot_digits(X_recovered[::2100])
plt.title("Compressed", fontsize=16)

plt.savefig("mnist_compression_plot")
{% endhighlight %}

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/digits_compression.png?raw=true)

This signal preserving/noise filtering property makes PCA a very useful feature selection routine—for example, rather than training a classifier on very high-dimensional data, you might instead train the classifier on the lower-dimensional representation, which will automatically serve to filter out random noise in the inputs.

# Example by Hand using Eigendecomposition

{% highlight python %} 
import numpy as np
import pandas as pd
{% endhighlight %}

Let’s say we have a dataset which is $d+1$ dimensional. Where $d$ could be thought as $X_{train}$ and 1 could be thought as $y_{train}$ (labels) in modern machine learning paradigm. So, $X_{train} + y_{train}$ makes up our complete train dataset.

So, after we drop the labels we are left with $d$ dimensional dataset and this would be the dataset we will use to find the principal components.

Let our data matrix X be the score of three students. In this example, we have $n=5$ and $d=3$.

{% highlight python %} 
X = np.array([[90, 60, 90],[90,90,30],[60,60,60],[60,60,90],[30,30,30]])
{% endhighlight %}

Compute the mean of every dimension of the whole dataset. We will have $3 \times 1$ array.

{% highlight python %} 
mean_X = np.mean(X, axis = 0)
#array([66., 60., 60.])
{% endhighlight %}

Let's center the data first:

{% highlight python %} 
X_centered = X - mean_X
# array([[ 252., -378.,  -18.,  162.,  -18.],
#        [-378., 1092.,  -48., -618.,  -48.],
#        [ -18.,  -48.,   12.,   42.,   12.],
#        [ 162., -618.,   42.,  372.,   42.],
#        [ -18.,  -48.,   12.,   42.,   12.]])
{% endhighlight %}

Then, we compute the unbiased covariance matrix (sometimes also called as the variance-covariance matrix) associated with the data.

$$
\Sigma_i = \left[
\begin{array}{cc}
\sigma_{11}^2 & \sigma_{12}^2 & \sigma_{13}^2\\
\sigma_{21}^2 & \sigma_{22}^2 & \sigma_{23}^2\\
\sigma_{31}^2 & \sigma_{32}^2 & \sigma_{33}^2\\
\end{array} \right] 
$$

So, we can compute the covariance of two variables $X$ and $Y$ using the following formula

$$
cov(X, Y) = \frac{1}{n-1} \sum_{i=1}^{n} (X_{i} - \bar{X})(Y_{i} - \bar{Y})
$$

Using the above formula, we can find the covariance matrix of A. Also, the result would be a square matrix of $d\times d$ dimensions.

For this example, we will have $3 \times 3$ square matrix.

{% highlight python %} 
cov_mat = np.cov(X_centered, rowvar=False, bias=False)

#NOTE

#If `rowvar` is True (default), then each row represents a variable, with observations in the columns. 
#Otherwise, the relationship is transposed: each column represents a variable, while the rows contain observations.

#Default normalization (False) is by ``(N - 1)``, where ``N`` is the number of observations given (unbiased estimate). 
#If `bias` is True, then normalization is by ``N``. These values can be overridden by using
#the keyword ``ddof`` in numpy versions >= 1.5.

# array([[630., 450., 225.],
#        [450., 450.,   0.],
#        [225.,   0., 900.]])

#When using centered data, the result will be equal to X_centered.T @ X / (4), which comes from the fact that X.T X / (n-1)
{% endhighlight %}

**Compute Eigenvectors and corresponding Eigenvalues**

Intuitively, an eigenvector is a vector whose direction remains unchanged when a linear transformation is applied to it.

Now, we can easily compute eigenvalue and eigenvectors from the covariance matrix that we have above.

Let $A$ be a square matrix, $v$ a vector and $u$ a scalar that satisfies $Av = uv$, then $u$ is called eigenvalue associated with eigenvector $v$ of $A$.

The eigenvalues of $A$ are roots of the characteristic equation

$$
det (A - uI) = 0
$$

{% highlight python %} 
# eigenvectors and eigenvalues for the from the covariance matrix
# (X.T * X)/(n−1) is equal to the empirical covariance if the columns of X are mean-centered.
#(1)
eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat*4)
#the column ``eig_vec_cov[:,i]`` is the eigenvector corresponding to the eigenvalue ``eig_val_cov[i]``.
    
######################################################################
#You can also get eigenvalues by SVD composition of X_centered
#(2)
U, s, Vt = np.linalg.svd(X_centered, full_matrices=True)
print(s)
#[67.4562804  56.08522028 14.96991321]

print(np.square(s))
#array([4550.34976521, 3145.55193338,  224.09830141])

#(1) AND (2) ARE THE SAME!
# This comes from the fact that calculating the SVD of X consists of finding the eigenvalues and eigenvectors of XTX and XXT.
######################################################################  
    
# eig_val_cov
# array([ 224.09830141, 4550.34976521, 3145.55193338])

# eig_vec_cov
# array([[ 0.6487899 , -0.65580225, -0.3859988 ],
#        [-0.74104991, -0.4291978 , -0.51636642],
#        [-0.17296443, -0.62105769,  0.7644414 ]])
{% endhighlight %}

Sort the eigenvectors by decreasing eigenvalues and choose $k$ eigenvectors with the largest eigenvalues to form a $d \times k$ dimensional matrix $W$.

We started with the goal to reduce the dimensionality of our feature space, i.e., projecting the feature space via PCA onto a smaller subspace, where the eigenvectors will form the axes of this new feature subspace. So, in order to decide which eigenvector(s) we want to drop for our lower-dimensional subspace, we have to take a look at the corresponding eigenvalues of the eigenvectors. Roughly speaking, the eigenvectors with the lowest eigenvalues bear the least information about the distribution of the data, and those are the ones we want to drop.

The common approach is to rank the eigenvectors from highest to lowest corresponding eigenvalue and choose the top k eigenvectors.

So, after sorting the eigenvalues in decreasing order, we have:

{% highlight python %} 
sorted_eigenvalues = np.sort(eig_val_cov)
# array([ 224.09830141, 3145.55193338, 4550.34976521])
{% endhighlight %}

You can also compute the explained variance ratio using eigenvalues

{% highlight python %} 
variances = sorted_eigenvalues / sorted_eigenvalues.sum()
# array([0.57453911, 0.39716565, 0.02829524])

cumsum = np.cumsum(variances)
# array([0.57453911, 0.97170476, 1.        ])
{% endhighlight %}

The first two components explains 97.17% of the variance

For our simple example, where we are reducing a 3-dimensional feature space to a 2-dimensional feature subspace, we are combining the two eigenvectors with the highest eigenvalues to construct our $d \times k$ dimensional eigenvector matrix $W$.

So, eigenvectors corresponding to two maximum eigenvalues are :

{% highlight python %} 
W = np.vstack([eig_vec_cov[:,1], eig_vec_cov[:,2]]).T
# array([[-0.65580225, -0.3859988 ],
#        [-0.4291978 , -0.51636642],
#        [-0.62105769,  0.7644414 ]])
{% endhighlight %}

Transform the samples onto the new subspace:

{% highlight python %} 
y = X_centered @ W
# array([[-34.37098481,  13.66927088],
#        [ -9.98345733, -47.68820559],
#        [  3.93481353,   2.31599277],
#        [-14.69691716,  25.24923474],
#        [ 55.11654576,   6.45370719]])
{% endhighlight %}

In order to make sure that we have not made a mistake in our step by step approach, we will use another library that doesn’t rescale the input data by default.

Here, we will use the `PCA` class from the scikit-learn machine-learning library. For our convenience, we can directly specify to how many components we want to reduce our input dataset via the `n_components` parameter.

{% highlight python %} 
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
X2D = pca.fit_transform(X)

pca.explained_variance_
#array([1137.5874413 ,  786.38798335])

pca.explained_variance_ratio_
#array([0.57453911, 0.39716565])

X2D 
# array([[-34.37098481, -13.66927088],
#        [ -9.98345733,  47.68820559],
#        [  3.93481353,  -2.31599277],
#        [-14.69691716, -25.24923474],
#        [ 55.11654576,  -6.45370719]]) 
{% endhighlight %}

Notice that running PCA multiple times on slightly different datasets may result in different results. In general the only difference is that some axes may be flipped. In this example, PCA using Scikit-Learn gives the same projection as the one given by the SVD approach, except both axes are flipped:

## WHY PCA WORKS?

PCA is a method that brings together:

1. A measure of how each variable is associated with one another. (Covariance matrix.)
2. The directions in which our data are dispersed. (Eigenvectors represent directions.)
3. The relative importance of these different directions. (Eigenvalues represent magnitude, or importance.)

PCA combines our predictors and allows us to drop the eigenvectors that are relatively unimportant.

---
layout: post
title: "Support Vector Machines in theoretical details"
author: "MMA"
comments: true
---

NOTE: This blog post consists of images. It might take a while to load in your browser!

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-01.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-02.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-03.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-04.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-05.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-06.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-07.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-08.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-09.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/slater_saddle.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-10.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/duality_gap.jpeg?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-11.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-12.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-13.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-14.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-15.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-16.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-17.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-18.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-19.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-20.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-21.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/svm_soft_margin_geometric.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-22.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/c_biasvariance.jpeg?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-23.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-24.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-25.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-26.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-27.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-28.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/DOC101719-10172019084921-29.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/linear_rbf.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/why_sv_quadratic.jpeg?raw=true)


# Visual Representation of Kernels

A kernelized SVM is equivalent to a linear SVM that operates in feature space rather than input space. Conceptually, you can think of this as mapping the data (possibly nonlinearly) into feature space, then using a linear SVM. However, the actual steps taken when using a kernelized SVM don't look like this because the kernel trick is used. Rather than explicitly mapping the data into feature space, the mapping is defined implicitly by the kernel function, which returns the dot product between feature space representations of two data points. 

An intuitive view of Kernels would be that they correspond to functions that measure how closely related vectors $x$ and $z$ are. So when $x$ and $z$ are similar the Kernel will output a large value, and when they are dissimilar $K$ will be small. Knowing this justifies the use of the Gaussian Kernel as a measure of similarity

$$
K(x,z) = \exp[ \left( - \frac{||x-z||^2}{2 \sigma^2}\right)
$$

because the value is close to 1 when they are similar and close to 0 when they are not. When using a Kernel in a linear model, it is just like transforming the input data, then running the model in the transformed space.

Let's give some visual examples using sum of polynomials:

$$
\phi(x_1, x_2) = (z_1,z_2,z_3) = (x_1,x_2, x_1^2 + x_2^2)
$$

```python
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from IPython.display import HTML, Image
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
%matplotlib inline
sns.set()

from sklearn.datasets import make_circles
X, y = make_circles(100, factor=.1, noise=.1)

def feature_map_1(X):
    return np.asarray((X[:,0], X[:,1], X[:,0]**2 + X[:,1]**2)).T

Z = feature_map_1(X)

#2D scatter plot
fig = plt.figure(figsize = (16,8))
ax = fig.add_subplot(1, 2, 1)
ax.scatter(X[:,0], X[:,1], c = y, cmap = 'viridis')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Original dataset')

#3D scatter plot
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter3D(Z[:,0],Z[:,1], Z[:,2],c = y, cmap = 'viridis' ) #,rstride = 5, cstride = 5, cmap = 'jet', alpha = .4, edgecolor = 'none' )
ax.set_xlabel('$z_1$')
ax.set_ylabel('$z_2$')
ax.set_zlabel('$z_3$')
ax.set_title('Transformed dataset')
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-05-20%20at%2007.43.15.png?raw=true)

Another example is a Gaussian Radial Basis Function (RBF) centered at 0,0:

$$
\phi(x_1, x_2) = (z_1,z_2,z_3) = (x_1,x_2, e^{- [x_1^2 + x_2^2]  })
$$

```python
def feature_map_2(X):  
    return np.asarray((X[:,0], X[:,1], np.exp( -( X[:,0]**2 + X[:,1]**2)))).T

#Generate dataset and feature-map
X, y = make_circles(100, factor=.1, noise=.1)
Z = feature_map_2(X)

#2D scatter plot
fig = plt.figure(figsize = (16,8))
ax = fig.add_subplot(1, 2, 1)
ax.scatter(X[:,0], X[:,1], c = y, cmap = 'viridis')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Original dataset')

#3D scatter plot
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter3D(Z[:,0],Z[:,1], Z[:,2],c = y, cmap = 'viridis' ) #,rstride = 5, cstride = 5, cmap = 'jet', alpha = .4, edgecolor = 'none' )
ax.set_xlabel('$z_1$')
ax.set_ylabel('$z_2$')
ax.set_zlabel('$z_3$')
ax.set_title('Transformed dataset')
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-05-20%20at%2007.43.58.png?raw=true)

Another Polynomial kernel...

$$
K(\mathbf{x},\mathbf{x'}) = (\mathbf{x}^T\mathbf{x'})^d
$$

Let $d=2$ and $x = (x1,x2)^{T}$ we get:
    
$$
\begin{split}
k=\left(\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}, \begin{pmatrix} x_1' \\ x_2' \end{pmatrix} \right) & = (x_1x_2' + x_2x_2')^2\\
& = 2x_1x_1'x_2x_2' + (x_1x_1')^2 + (x_2x_2')^2\\
& = (\sqrt{2}x_1x_2  \ x_1^2 \ x_2^2) \ \begin{pmatrix} \sqrt{2}x_1'x_2' \\ x_1'^2  \\ x_2'^2 \end{pmatrix}\\
&= \phi(\mathbf{x})^T \phi(\mathbf{x'})
\end{split}
$$

where 

$$
\phi(\mathbf{x}) = \phi\left(\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}\right)=\begin{pmatrix}\sqrt{2}x_1x_2 \\ x_1^2 \\ x_2^2 \end{pmatrix}
$$

In the plot of the transformed data we map

$$
x_1, x_2 : \rightarrow z_1, z_2, z_3
$$

$$
z_1 = \sqrt{2}x_1x_2 \ \   z_2 = x_1^2 \ \  z_3 = x_2^2
$$

This time, let's explicitly go to high-dimensional feature space and apply a linear SVM to plot the decision boundary:

```python
def feature_map_3(X):  
    return np.asarray(( np.sqrt(2) *X[:,0] * X[:,1], X[:,0]**2, X[:,1]**2)).T

X, y = make_circles(100, factor=.1, noise=.1, random_state = 0)
Z = feature_map_3(X)

#2D scatter plot
fig = plt.figure(figsize = (16,8))
ax = fig.add_subplot(1, 2, 1)
ax.scatter(X[:,0], X[:,1], c = y, cmap = 'viridis')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Original data')

#3D scatter plot
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.scatter3D(Z[:,0],Z[:,1], Z[:,2],c = y, cmap = 'viridis' ) #,rstride = 5, cstride = 5, cmap = 'jet', alpha = .4, edgecolor = 'none' )
ax.set_xlabel('$z_1$')
ax.set_ylabel('$z_2$')
ax.set_zlabel('$z_3$')
ax.set_title('Transformed data: ')

#Let's use this converted data and apply linear SVM in higher dimensions.
#SVM using kernel 3 - feature map 3
clf = SVC(C = 1, kernel = 'linear')
clf.fit(Z, y) 

w = clf.coef_.flatten()
b = clf.intercept_.flatten()
print('w=',w,'b=',b)

# create x,y
xx, yy = np.meshgrid(np.linspace(-1,1), np.linspace(0,1))

# calculate corresponding z
boundary = (-w[0] * xx - w[1] * yy - b) * 1. /w[2]


# plot the surface

ax.plot_surface(xx, yy, boundary, alpha = .3)
ax.set_ylim(.2,1.2)
ax.set_zlim(-.9,1.1)
#ax.view_init(0, 260)

plt.show()
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-05-20%20at%2007.45.47.png?raw=true)

# Sklearn implementation using custom Kernel

```python
def my_kernel_1(X,Y):
    return np.dot(feature_map_1(X),feature_map_1(Y).T )

def my_kernel_2(X,Y):
    return np.dot(feature_map_2(X),feature_map_2(Y).T )

def my_kernel_3(X,Y):
    return np.dot(feature_map_3(X),feature_map_3(Y).T )

clf = SVC(kernel=my_kernel_1)
clf.fit(X, y) 

# predict on training examples - print accuracy score
print('Accuracy score using feature map n1',accuracy_score(y, clf.predict(X)))
#Accuracy score using feature map n1 1.0

#SVM using kernel 2 - feature map 2
clf = SVC(kernel=my_kernel_2)
clf.fit(X, y) 

# predict on training examples - print accuracy score
print('Accuracy score using feature map n2',accuracy_score(y, clf.predict(X)))
#Accuracy score using feature map n2 1.0

#SVM using kernel 3 - feature map 3
clf = SVC(kernel=my_kernel_3)
clf.fit(X, y) 

# predict on training examples - print accuracy score
print('Accuracy score using feature map n3',accuracy_score(y, clf.predict(X)))
#Accuracy score using feature map n3 1.0
```

Let's plot the decision boundary for the third kernel to see whether it is working or not:

```python
clf = SVC(kernel=my_kernel_3, C = 1)
# kernel computation
clf.fit(X, y) 

#Initialize data
h = .01 #Stepsize in the mesh
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#Predict on meshgrid
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize = (7,7))
plt.contourf(xx, yy, Z,1, colors = ['darkblue','yellow'], alpha = .1)
plt.contour(xx, yy, Z, cmap = 'viridis')

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors = 'k')
plt.title('Support Vector Machine with polynomial'
          ' kernel')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
```

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202020-05-20%20at%2007.46.53.png?raw=true)

# Sklearn implementation of SVC with Gram matrix

Consider a dataset of $m$ data points which are $n$ dimensional vectors $\in R^{n}$, the gram matrix is the $m \times m$ matrix for which each entry is the kernel between the corresponding data points.

$$
G_{i,j} = K(x^{(i)}, x^{(j)})
$$

```python
clf = SVC(kernel='precomputed')
# kernel computation
gram = np.dot(feature_map_2(X), feature_map_2(X).T)
clf.fit(gram, y) 

# prediction errors on training examples
np.sum(y - clf.predict(gram))
#0
```

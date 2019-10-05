---
layout: post
title: "Univariate/Multivariate Gaussian Distribution and their properties"
author: "MMA"
comments: true
---

# Univariate Normal Distribution

The normal distribution, also known as Gaussian distribution, is defined by two parameters, mean $\mu$, which is expected value of the distribution and standard deviation $\sigma$ which corresponds to the expected squared deviation from the mean. Mean, $\mu$ controls the Gaussian's center position and the standard deviation controls the shape of the distribution. The square of standard deviation is typically referred to as the variance $\sigma^{2}$. We denote this distribution as $N(\mu, \sigma^{2})$.

Given the mean  and variance, one can calculate probability distribution function of normal distribution with a normalised Gaussian function for a value $x$, the density is:

$$
P(x \mid \mu, \sigma^{2}) = \frac{1}{\sqrt{2\pi \sigma^{2}}} exp \left(- \frac{(x - \mu)^{2}}{2\sigma^{2}} \right)
$$

We call this distribution univariate because it consists of one random variable.

{% highlight python %} 
# Load libraries
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def univariate_normal(x, mean, variance):
    """pdf of the univariate normal distribution."""
    return ((1. / np.sqrt(2 * np.pi * variance)) * 
            np.exp(-(x - mean)**2 / (2 * variance)))

# Plot different Univariate Normals
x = np.linspace(-3, 5, num=150)
fig = plt.figure(figsize=(5, 3))
plt.plot(
    x, univariate_normal(x, mean=0, variance=1), 
    label="$N(0, 1)$")
plt.plot(
    x, univariate_normal(x, mean=2, variance=3), 
    label="$n(2, 3)$")
plt.plot(
    x, univariate_normal(x, mean=0, variance=0.2), 
    label="$n(0, 0.2)$")
plt.xlabel('$x$', fontsize=13)
plt.ylabel('density: $p(x)$', fontsize=13)
plt.title('Univariate normal distributions')
plt.ylim([0, 1])
plt.xlim([-3, 5])
plt.legend(loc=1)
fig.subplots_adjust(bottom=0.15)
plt.savefig('univariate_normal_distribution.png')
plt.show()
{% endhighlight %}

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/univariate_normal_distribution.png?raw=true)

# Multivariate Normal Distribution

The multivariate normal distribution is a multidimensional generalisation of the one dimensional normal distribution. It represents the distribution of a multivariate random variable, that is made up of multiple random variables which can be correlated with each other.

Like the univariate normal distribution, the multivariate normal is defined by sets of parameters: the mean _vector_ $\mu$. which is expected value of the distribution and the variance-covariance matrix $\Sigma$, which measures how two random variables depend on each other and how they change together.

We denote the covariance between variables $X$ and $Y$ as $Cov(X,Y)$.

The multivariate normal with dimensionality $d$ has a joint probability density given by:

$$
P(\mathbf{x} \mid \mu, \Sigma) = \frac{1}{\sqrt{2(\pi)^{d} \lvert \Sigma \rvert}} exp \left(- \frac{1}{2} (\mathbf{x} - \mu)^{T} \Sigma^{-1}  (\mathbf{x} - \mu)\right)
$$

where $\mathbf{x}$ is a random vector of size $d$, $\mu$ is $d \times 1$ mean vector and $\Sigma$ is the (symmetric and positive definite) covariance matrix of size $d \times d$ and $\lvert \Sigma \rvert$ is the determinant. We denote this multivariate normal distribution as $N(\mu, \Sigma)$. 

{% highlight python %} 
def multivariate_normal(x, d, mean, covariance):
    """pdf of the multivariate normal distribution."""
    x_m = x - mean
    return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) * 
            np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))
{% endhighlight %}

Plotting a multivariate distribution of more than two variables might be hard. Therefore, let's give an example of bivariate normal distribution. 

Assume a 2-dimensional random vector.

$$
\mathbf{x} = \begin{bmatrix}x_{1}\\ x_{2}\end{bmatrix}
$$

has a normal distribution $N(\mu, \Sigma)$ where

$$
\mu = \begin{bmatrix} \mu_{1} \\ \mu_{2} \end{bmatrix}
$$

and

$$
\Sigma = \begin{bmatrix}\Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22}\end{bmatrix}
$$

* If $x_{1}$ and $x_{2}$ is independent, covariance between $x_{1}$ and $x_{1}$ is set to zero, for instance,

  $$
N\left( \begin{bmatrix}0\\ 1\end{bmatrix}, \begin{bmatrix}1 & 0 \\ 0 & 1 \end{bmatrix}\right)
$$

* If $x_{1}$ and $x_{2}$ is set to be different than 0, we can say that both variable are correlated, for instance,

  $$
N\left( \begin{bmatrix}0\\ 1\end{bmatrix}, \begin{bmatrix}1 & 0.8 \\ 0.8 & 1 \end{bmatrix}\right)
$$

  meaning that increasing $x_{1}$ will increase the probability that $x_{2}$ will also increase.

Note that covariance matrix must be positive definite.

{% highlight python %} 
np.linalg.eigvals(bivariate_covariance)
#array([1.8, 0.2])
#We see that this covariance matrix is indeed positive definite (see The Spectral Theorem for Matrices).
{% endhighlight %}

Let's plot multivariate normal distribution for both cases:

{% highlight python %} 
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

def multivariate_normal(x, d, mean, covariance):
    """pdf of the multivariate normal distribution."""
    x_m = x - mean
    return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) * 
            np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))

# Plot bivariate distribution
def generate_surface(mean, covariance, d):
    """Helper function to generate density surface."""
    nb_of_x = 100 # grid size
    x1s = np.linspace(-5, 5, num=nb_of_x)
    x2s = np.linspace(-5, 5, num=nb_of_x)
    x1, x2 = np.meshgrid(x1s, x2s) # Generate grid
    pdf = np.zeros((nb_of_x, nb_of_x))
    # Fill the cost matrix for each combination of weights
    for i in range(nb_of_x):
        for j in range(nb_of_x):
            pdf[i,j] = multivariate_normal(
                np.matrix([[x1[i,j]], [x2[i,j]]]), 
                d, mean, covariance)
    return x1, x2, pdf  # x1, x2, pdf(x1,x2)

# subplot
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
d = 2  # number of dimensions

# Plot of independent Normals
bivariate_mean = np.matrix([[0.], [0.]])  # Mean
bivariate_covariance = np.matrix([
    [1., 0.], 
    [0., 1.]])  # Covariance
x1, x2, p = generate_surface(
    bivariate_mean, bivariate_covariance, d)
# Plot bivariate distribution
con = ax1.contourf(x1, x2, p, 100, cmap='rainbow')
ax1.set_xlabel('$x_1$', fontsize=13)
ax1.set_ylabel('$x_2$', fontsize=13)
ax1.axis([-2.5, 2.5, -2.5, 2.5])
ax1.set_aspect('equal')
ax1.set_title('Independent variables', fontsize=12)

# Plot of correlated Normals
bivariate_mean = np.matrix([[0.], [1.]])  # Mean
bivariate_covariance = np.matrix([
    [1., 0.8], 
    [0.8, 1.]])  # Covariance
x1, x2, p = generate_surface(
    bivariate_mean, bivariate_covariance, d)
# Plot bivariate distribution
con = ax2.contourf(x1, x2, p, 100, cmap='rainbow')
ax2.set_xlabel('$x_1$', fontsize=13)
ax2.set_ylabel('$x_2$', fontsize=13)
ax2.axis([-2.5, 2.5, -1.5, 3.5])
ax2.set_aspect('equal')
ax2.set_title('Correlated variables', fontsize=12)

# Add colorbar and title
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
cbar = fig.colorbar(con, cax=cbar_ax)
cbar.ax.set_ylabel('$p(x_1, x_2)$', fontsize=13)
plt.suptitle('Bivariate normal distributions', fontsize=13, y=0.95)
plt.savefig('Bivariate_normal_distributon')
plt.show()
{% endhighlight %}

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Bivariate_normal_distributon.png?raw=true)

# Affine transformation of univariate normal distribution

Suppose $X \sim N(\mu, \sigma^{2})$ and $a, b \in \mathbb{R}$ with $a \neq 0. If we define an affine transformation $Y = g(X) = aX+b$, then $Y \sim N(a\mu + b, a^{2}\sigma^{2})$,

meaning that linear combination of independent random variables are normal.

For example, if $X \sim N(1, 2)$,
* $Y= X+ 3 \Rightarrow Y \sim N(4, 2)$
* $Y= 2X+ 3 \Rightarrow Y \sim N(5, 8)$

Let's try to prove this. Using the inverse transformation method,

$$
\begin{split}
F_{Y}(y) = P(Y \leq y) &= P(aX+b \leq y)\\
&= P(X \leq \frac{Y-b}{a})\\
&= \int_{-\infty}^{\frac{y-b}{a}}\frac{1}{\sqrt{2\pi \sigma^{2}}} exp \left(- \frac{(x - \mu)^{2}}{2\sigma^{2}} \right) dx
\end{split}
$$

Therefore, if we take first-order derivative of this CDF, we will get probability density function of $Y$:

$$
\begin{split}
f_{Y}(y) &= \frac{d}{dy} \int_{-\infty}^{\frac{y-b}{a}}\frac{1}{\sqrt{2\pi \sigma^{2}}} exp \left(- \frac{(x - \mu)^{2}}{2\sigma^{2}} \right) dx\\
&= \frac{1}{\sqrt{2\pi \sigma^{2}}} exp \left(- \frac{\left(\left(\frac{y-b}{a} \right) - \mu\right)^{2}}{2\sigma^{2}} \right) \left(\frac{d}{dy}\frac{y-b}{a} \right)\\
&= \frac{1}{a\sigma \sqrt{2\pi}} exp \left(- \frac{\left(y-(a\mu +b)\right)^{2}}{2a^{2}\sigma^{2}} \right) \left(\frac{d}{dy}\frac{y-b}{a} \right)\\
\end{split}
$$

and then simply so $YY \sim N(a\mu + b, a^{2}\sigma^{2})$.

# Sampling from a multivarivate normal distribution

{% highlight python %} 
# Sample from:
d = 2 # Number of random variables
mean = np.matrix([[0.], [1.]])
covariance = np.matrix([
    [1, 0.8], 
    [0.8, 1]
])

# Create L
L = np.linalg.cholesky(covariance)
#matrix([[1. , 0. ],
#        [0.8, 0.6]])
    
# Sample X from standard normal
n = 50  # Samples to draw
X = np.random.normal(size=(d, n))
# shape of X is (2, 50)

# Apply the transformation
Y = L.dot(X) + mean
# shape of Y is (2, 50)

# Plot the samples and the distribution
fig, ax = plt.subplots(figsize=(6, 4.5))
# Plot bivariate distribution
x1, x2, p = generate_surface(mean, covariance, d)
con = ax.contourf(x1, x2, p, 100, cmap='rainbow')
# Plot samples
ax.plot(Y[0,:], Y[1,:], 'ro', alpha=.6,
        markeredgecolor='k', markeredgewidth=0.5)
ax.set_xlabel('$y_1$', fontsize=13)
ax.set_ylabel('$y_2$', fontsize=13)
ax.axis([-2.5, 2.5, -1.5, 3.5])
ax.set_aspect('equal')
ax.set_title('Samples from bivariate normal distribution')
cbar = plt.colorbar(con)
cbar.ax.set_ylabel('density: $p(y_1, y_2)$', fontsize=13)
plt.savefig('sampling_from_multivariate_normal_distribution')
plt.show()
{% endhighlight %}

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/sampling_from_multivariate_normal_distribution.png?raw=true)
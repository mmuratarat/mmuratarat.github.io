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

Suppose $X \sim N(\mu, \sigma^{2})$ and $a, b \in \mathbb{R}$ with $a \neq 0$. If we define an affine transformation $Y = g(X) = aX+b$, then $Y \sim N(a\mu + b, a^{2}\sigma^{2})$,

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
&= \frac{1}{a\sigma \sqrt{2\pi}} exp \left(- \frac{\left(y-(a\mu +b)\right)^{2}}{2a^{2}\sigma^{2}} \right)\\
\end{split}
$$

and then simply so $Y \sim N(a\mu + b, a^{2}\sigma^{2})$.

# Affine transformation of univariate normal distribution

It is possible to transform a multivariate normal distribution into a new normal distribution, using an affine transformation. More specifically, if $X$ is normally distributed and $Y= LX + u$ with $L$ is a linear transformation and $u$ is a vector. Then, $y$ is also norally distributed with mean $\mu_{Y} = u + L \mu_{X}$ and covariance matrix $\Sigma_{Y} = L \Sigma_{X} L^{T}$. 

$$
\begin{split}
Y \sim N(\mu_{Y}, \Sigma_{Y}) \quad\quad X \sim N(\mu_{X}, \Sigma_{X}) \\
N(\mu_{Y}, \Sigma_{Y}) = N(u + L\mu_{X}, L\Sigma_{X}L^T) = LN(\mu_{X}, \Sigma_{X}) + u
\end{split}
$$

This can be proven as follows:

$$
\mu_{Y} = \mathbb{E}[Y] = \mathbb{E}[LX + u] = \mathbb{E}[LX] + u = L\mu_{X} + u
$$

$$
\begin{split}
\Sigma_{Y} & = \mathbb{E}[(Y-\mu_{Y})(Y-\mu_{Y})^\top] \\
           & = \mathbb{E}[(LX+u - L\mu_{X}-u)(LX+u - L\mu_{X}-u)^\top] \\
           & = \mathbb{E}[(L(X-\mu_{X})) (L(X-\mu_{X}))^\top] \\
           & = \mathbb{E}[L(X-\mu_{X}) (X-\mu_{X})^\top L^\top] \\
           & = L\mathbb{E}[(X-\mu_{X})(X-\mu_{X})^\top]L^\top \\
           & = L\Sigma_{X}L^\top
\end{split}
$$

# Sampling from a multivarivate normal distribution

The previous formula helps us to sample from any multivariate Gaussian distribution.

in order to do this, we can sample $X$ from $N(0, I_{d})$ where mean is the vector $\mu =0$ and variance-covariance matrix is the identity matrix $\sigma_{X} = I_{d}$ (standard multivariate normal distribution). Sampling from this distribution is easier because each variable in $X$ is independent from all other variables, we can just sample each variable seperately.

It is then possible to sample $Y$ from $N(\mu_{Y}, \Sigma_{Y})$ by sampling $X$ and applying affine transformation on samples. This transform is $Y = LX + u$ where we know the covariance of Y is $\Sigma_{Y} = L \Sigma_{X} L^{T}$. Since $\Sigma_{X} = I_{d}$, we can write $\Sigma_{Y} = L \Sigma_{X} L^{T} = L I_{d} L^{T} = L L^{T}$. $L$ can be found by a technique called Cholesky decomposition. The vector $u$ is then $\mu_{Y}$ since $\mu_{X} = 0$ ($u = \mu_{Y} - L \mu_{X}$).

Lets try this out by sampling 50 samples from:


$$
Y
\sim
N\left(
\begin{bmatrix} 0 \\ 
1 
\end{bmatrix},
\begin{bmatrix}
1 & 0.8 \\
0.8 & 1
\end{bmatrix}\right)
$$

The sampling is done by the following code and the samples are plotted as red dots on the probability density surface below.

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

# Marginal normal distributions

If both $\mathbf{x}$ and $\mathbf{y}$ are jointly normal random vectors defined as:

$$
\begin{bmatrix}
\mathbf{x} \\
\mathbf{y} 
\end{bmatrix}
\sim
\mathcal{N}\left(
\begin{bmatrix}
\mu_{\mathbf{x}} \\
\mu_{\mathbf{y}}
\end{bmatrix},
\begin{bmatrix}
A & C \\
C^T & B
\end{bmatrix}
\right)
= \mathcal{N}(\mu, \Sigma)
, \qquad 
\Sigma^{-1} = \Lambda = 
\begin{bmatrix}
\tilde{A} & \tilde{C} \\
\tilde{C}^T & \tilde{B}
\end{bmatrix}
$$

{% highlight python %} 
d = 2  # dimensions
mean = np.matrix([[0.], [1.]])
cov = np.matrix([
    [1, 0.8], 
    [0.8, 1]
])
# Get the mean values from the vector
mean_x = mean[0,0]
mean_y = mean[1,0]
# Get the blocks (single values in this case) from 
#  the covariance matrix
A = cov[0, 0]
B = cov[1, 1]
C = cov[0, 1]  # = C transpose in this case
{% endhighlight %}

A marginal distribution is the distribution of a subset of random variables from the original distribution. It represents the probabilities or densities of the variables in the subset without reference to the other values in the original distribution.

In our case of the 2D multivariate normal the marginal distibutions are the univariate distributions of each component $\mathbf{x}$ and $\mathbf{y}$ seperately. They are defined as:

$$
\begin{split}
p(\mathbf{x}) & = \mathcal{N}(\mu_{\mathbf{x}}, A) \\
p(\mathbf{y}) & = \mathcal{N}(\mu_{\mathbf{y}}, B)
\end{split}
$$

{% highlight python %} 
#Load libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
%matplotlib inline

d = 2  # dimensions
mean = np.matrix([[0.], [1.]])
cov = np.matrix([
    [1, 0.8], 
    [0.8, 1]
])
# Get the mean values from the vector
mean_x = mean[0,0]
mean_y = mean[1,0]
# Get the blocks (single values in this case) from 
#  the covariance matrix
A = cov[0, 0]
B = cov[1, 1]
C = cov[0, 1]  # = C transpose in this case

def univariate_normal(x, mean, variance):
    """pdf of the univariate normal distribution."""
    return ((1. / np.sqrt(2 * np.pi * variance)) * 
            np.exp(-(x - mean)**2 / (2 * variance)))

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

# Plot the conditional distributions
fig = plt.figure(figsize=(7, 7))
gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])
# gs.update(wspace=0., hspace=0.)
plt.suptitle('Marginal distributions', y=0.93)

# Plot surface on top left
ax1 = plt.subplot(gs[0])
x, y, p = generate_surface(mean, cov, d)
# Plot bivariate distribution
con = ax1.contourf(x, y, p, 100, cmap='rainbow')
ax1.set_xlabel('$x$', fontsize=13)
ax1.set_ylabel('$y$', fontsize=13)
ax1.yaxis.set_label_position('right')
ax1.axis([-2.5, 2.5, -1.5, 3.5])

# Plot y
ax2 = plt.subplot(gs[1])
y = np.linspace(-5, 5, num=100)
py = univariate_normal(y, mean_y, A)
# Plot univariate distribution
ax2.plot(py, y, 'b--', label=f'$p(y)$')
ax2.legend(loc=0)
ax2.set_xlabel('density', fontsize=13)
ax2.set_ylim(-1.5, 3.5)

# Plot x
ax3 = plt.subplot(gs[2])
x = np.linspace(-5, 5, num=100)
px = univariate_normal(x, mean_x, B)
# Plot univariate distribution
ax3.plot(x, px, 'r--', label=f'$p(x_1)$')
ax3.legend(loc=0)
ax3.set_ylabel('density', fontsize=13)
ax3.yaxis.set_label_position('right')
ax3.set_xlim(-2.5, 2.5)

# Clear axis 4 and plot colarbar in its place
ax4 = plt.subplot(gs[3])
ax4.set_visible(False)
divider = make_axes_locatable(ax4)
cax = divider.append_axes('left', size='20%', pad=0.05)
cbar = fig.colorbar(con, cax=cax)
cbar.ax.set_ylabel('density: $p(x, xy)$', fontsize=13)
plt.savefig('marginal_normal_distributions')
plt.show()
{% endhighlight %}

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/marginal_normal_distributions.png?raw=true)

# What are the different types of covariance matrices?

Let's look at a few examples of covariance structures that we could specify for multivariate Gaussian distribution:

{% highlight python %} 
#Load libraries
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from scipy.stats import multivariate_normal

x, y = np.mgrid[-4:4:.01, -4:4:.01]
position = np.empty(x.shape + (2,))
position[:, :, 0] = x
position[:, :, 1] = y

# different values for the covariance matrix
covariances = [ [[1, 0], [0, 1]], [[1, 0], [0, 3]], [[1, -1], [-1, 3]] ]
titles = ['spherical', 'diag', 'full']

plt.figure(figsize = (15, 6))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    z = multivariate_normal([0, 0], covariances[i]).pdf(position)
    plt.contour(x, y, z)
    plt.title('{}, {}'.format(titles[i], covariances[i]))
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])

plt.show()
{% endhighlight %}

One way to view a Gaussian distribution in two dimensions is what's called a contour plot. The coloring represents the region's intensity, or how high it was in probability. 

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/types_covariance_matrix.png?raw=true)

In the plot above, the center area that has dark red color is the region of highest probability, while the blue area corresponds to a low probability.

The first plot is refered to as a Spherical Gaussian, since the probability distribution has spherical (circular) symmetry. The covariance matrix is a diagonal covariance with equal elements along the diagonal. By specifying a diagonal covariance, what we're seeing is that there's no correlation between our two random variables, because the off-diagonal correlations takes the value of 0. You can simply interpret it as there is no linear relationship exists between variables. However, note that this does not necessarily mean that they are independent. Furthermore, by having equal values of the variances along the diagonal, we end up with a circular shape to the distribution because we are saying that the spread along each one of these two dimensions is exactly the same.

In contrast, the middle plot's covariance matrix is also a diagonal one, but we can see that if we were to specify different variances along the diagonal, then the spread in each of these dimensions is different and so what we end up with are these axis-aligned ellipses. This is refered to as a Diagonal Gaussian.

Finally, we have the Full Gaussian. A full covariance matrix allows for correlation between our two random variables (non zero off diagonal value) we can provide these non-axis aligned ellipses. So in this example that we're showing here, these two variables are negatively correlated, meaning if one variable is high, it's more likely that the other value is low.

# Maximum Likelihood Estimation for Univariate Gaussian Distribution

Given the assumption that the observations from the sample are i.i.d., the likelihood function can be written as:

$$
\begin{split}
L(\mu, \sigma^{2}; x_{1}, x_{2}, \ldots , x_{n}) &= \prod_{i=1}^{n} f_{X}(x_{i} ; \mu , \sigma^{2})\\
&= \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi \sigma^{2}}} exp \left(- \frac{(x_{i} - \mu)^{2}}{2\sigma^{2}} \right)\\
&= \left(2\pi \sigma^{2} \right)^{-n/2} exp \left(-\frac{1}{2\sigma^{2}} \sum_{i=1}^{n}(x_{i} - \mu)^{2} \right)\\
\end{split}
$$

The log-likelihood function is

$$
l(\mu, \sigma^{2}; x_{1}, x_{2}, \ldots , x_{n}) = \frac{-n}{2}ln(2\pi) -\frac{n}{2} ln(\sigma^{2})-\frac{1}{2\sigma^{2}}\sum_{i=1}^{n}(x_{i} - \mu)^{2} 
$$

If we take first-order partial derivatives of the log-likelihood function with respect to the mean $\mu$ and variance $\sigma^{2}$ and set the equations zero and solve them, we will have the maximum likelihood estimators of the mean and the variance, which are:

$$
\mu_{MLE} = \frac{1}{n} \sum_{i=1}^{n} x_{i}
$$

and 

$$
\sigma_{MLE}^{2} = \frac{1}{n} \sum_{i=1}^{n} \left(x_{i} - \mu_{MLE}\right)^{2}
$$

respectively. Thus, the estimator $\mu_{MLE}$ is equal to the sample mean and the estimator $\sigma_{MLE}^{2}$ is equal to the unadjusted sample variance.

# Maximum Likelihood Estimation for Multivariat Gaussian Distribution

The maximum likelihood estimators of the mean $\mu$ and the variance $\Sigma$ for multivariate normal distribution are found similarly and are as follows:

$$
\mu_{MLE} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{x}_{i}
$$

and 

$$
\Sigma_{MLE} = \frac{1}{n} \sum_{i=1}^{n} \left(\mathbf{x}_{i} - \mu_{MLE}\right) \left(\mathbf{x}_{i} - \mu_{MLE}\right)^{T}
$$

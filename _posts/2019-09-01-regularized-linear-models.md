---
layout: post
title: "Regularized Linear Models"
author: "MMA"
comments: true
---
Ridge regression and the Lasso are two forms of regularized regression (i.e., to constraint the model) which are typically achieved by constraining the weights of the model. These methods are seeking to alleviate the consequences of multicollinearity and overfitting the training set (by reducing model complexity).

When two (or multiple) features are fully linearly dependent, we have singular (noninvertible) $\mathbf{X}^{T} \cdot \mathbf{X}$ since $\mathbf{X}^{T} \cdot \mathbf{X}$ is not full rank (_rank deficiency_). This is obviously going to lead to problems because since $\mathbf{X}^{T} \cdot \mathbf{X}$ is not invertible, we cannot compute $\hat{\theta}_{OLS} = \left(\mathbf{X}^{T} \cdot \mathbf{X} \right)^{-1} \cdot \mathbf{X}^{T}y$. Actually we can, however, estimates of coefficients will be unrealistically large, untrustable / unstable, meaning that if you construct estimators from different data samples you will potentially get wildly different estimates of your coefficient values. Similarly, the variance of the estimates,

$$
Var(\hat{\theta}_{OLS}) = \sigma^{2} \left(\mathbf{X}^{T} \cdot \mathbf{X} \right)^{-1}
$$

will also blow up when $\mathbf{X}^{T} \cdot \mathbf{X}$ is singular. If that matrix isn’t exactly singular, but is
close to being non-invertible, the variances will become huge. 

If you dive into the matrix algebra, you will find that the term $\mathbf{X}^{T} \cdot \mathbf{X}$ is equal to a matrix with ones on the diagonals and the pairwise Pearson’s correlation coefficients ($\rho$) on the off-diagonals (this is the true when columns are standardized.):

$$
(\mathbf{X}^{T} \cdot \mathbf{X}) =\begin{bmatrix} 1 & \rho \\ \rho & 1 \end{bmatrix}
$$

As the correlation values increase, the values within $(\mathbf{X}^{T} \cdot \mathbf{X})^{-1}$ also increase. Even with a low residual variance, multicollinearity can cause large increases in estimator variance. Here are a few examples of the effect of multicollinearity using a hypothetical regression with two predictors:

$$
\begin{split}
 \rho = .3 &\rightarrow (\mathbf{X}^{T} \cdot \mathbf{X})^{-1} =\begin{bmatrix} 1 & \rho \\ \rho & 1 \end{bmatrix}^{-1} = \begin{bmatrix} 1.09 & -0.33 \\ -0.33 & 1.09 \end{bmatrix}\\
 \rho = .9 &\rightarrow (\mathbf{X}^{T} \cdot \mathbf{X})^{-1} =\begin{bmatrix} 1 & \rho \\ \rho & 1 \end{bmatrix}^{-1} = \begin{bmatrix} 5.26 & -4.73 \\ -5.26 & -4.73 \end{bmatrix}\\
 \rho = .999 &\rightarrow (\mathbf{X}^{T} \cdot \mathbf{X})^{-1} =\begin{bmatrix} 1 & \rho \\ \rho & 1 \end{bmatrix}^{-1} = \begin{bmatrix} 500.25 & -499.75 \\ -499.75 & 500.25\end{bmatrix}
\end{split}
$$

Large estimator variance also undermines the trustworthiness of hypothesis testing of the significance of coefficients. Because, consequently, corresponding t-statistics are typically lower:

$$
t_{n-2} = \frac{\hat{\beta_{j}} - 0}{s_{\beta_{j}}}
$$

An estimator with an inflated standard deviation, $s_{\beta_{j}}$, will thus yield a lower t-value,  which could lead to the false rejection of a significant predictor (ie. a type II error).

### Singular matrix 

A square matrix with no inverse is called singular (noninvertible). A matrix $A$ is singular if and only if $det(A) = 0$, which its determinant equals the product of the eigenvalues of $A$. In case of singularity, one of the eigenvalues is zero. In order to see this, you can consider the spectral decomposition of $A$. The spectral decomposition recasts a matrix in terms of its eigenvalues and eigenvectors. This representation turns out to be enormously useful:

Let $A$ be a real symmetric $p \times p$ matrix with eigenvalues $u_{1}, u_{2}, ..., u_{p}$ and corresponding orthonormal eigenvectors $v_{1}, v_{2},...,v_{p}$. Then,

$$
A = \sum_{i=1}^{p} u_{i}v_{i}v_{i}^{T}
$$

Inverse of A is then given by,

$$
A^{-1} = \sum_{i=1}^{p} u_{i}^{-1}v_{i}v_{i}^{T}
$$

Then, if one of the eigenvalues is zero, i.e., $u_{i} = 0$ for any $i$, $A^{-1}$ will be singular (noninvertible) since $\frac{1}{0}$ is undefined.

This comes from the fact that a matrix $A$ has an eigenvalue $u$ if and only if $A^{-1}$ has eigenvalue $u^{-1}$. To see this, note that $Av = u v \implies A^{-1}Av = u A^{-1}v\implies A^{-1}v = \frac{1}{u}v$ (When $A$ is multiplied by $A^{-1}$ the result is the identity matrix $I$).

In case of singular $\mathbf{X}^{T} \cdot \mathbf{X}$, its inverse $(\mathbf{X}^{T} \cdot \mathbf{X})^{-1}$ is not defined. Consequently, the OLS estimator $\hat{\theta}_{OLS} = \left(\mathbf{X}^{T} \cdot \mathbf{X} \right)^{-1} \cdot \mathbf{X}^{T}y$ does not exist. This happens in high-dimensional data, i.e. when the number of features exceeds the number of observations. An ad-hoc solution is to use Ridge or Lasso regression models. 

## An Example
Let's assume that

$$
\mathbf{X} = \begin{bmatrix}
1 & -1 & 2\\
1 & 0 & 1 \\
1 & 2 & -1 \\
1 & 1 & 0
\end{bmatrix}
$$

then,

$$
\mathbf{X}^{T} \cdot \mathbf{X} = \begin{bmatrix}
4 & 2 & 2\\
2 & 6 & -4 \\
2 & -4 & 6 \\
\end{bmatrix}
$$

which has eigenvalues equal to 10, 6 and 0. It's rank is 2 which is not full rank (3 - For a square matrix, the matrix is full rank if all rows and columns are linearly independent) and its determinant is zero (determinant is the product of eigenvalues). Then, this matrix is singular (non-invertible).

{% highlight python %}
import numpy as np

a = np.array([[4, 2, 2], [2, 6 , -4], [2, -4, 6]])

#Determinant
np.linalg.det(a)
#0.0

#Rank
np.linalg.matrix_rank(a)
#2
{% endhighlight %}

With the "ridge-fix", we get e.g.:

$$
\mathbf{X}^{T} \cdot \mathbf{X} + I = \begin{bmatrix}
5 & 2 & 2\\
2 & 7 & -4 \\
2 & -4 & 7 \\
\end{bmatrix}
$$

where $I$ is the identity matrix which we will see below in details. This matrix now has eigenvalues equal to 11, 7 and 1, meaning thats its determinant is not zero, thereby, it is invertible and we can compute the estimator of $\theta$.

But first, let's fresh our memories by shortly talking about Linear Regression and how to obtain the estimation of parameters vector using OLS.

# Linear Regression

Given a training data with $m$ observations and $n$ predictors (features/attributes/covariates), we try to fit each training example using the linear model

$$
\hat{y}^{(i)} = h_{\theta}(\mathbf{x}^{(i)}) = \theta^{T}x^{(i)} = \begin{bmatrix}\theta_{0} & \theta_{1} & \cdots & \theta_{n} \end{bmatrix} \begin{bmatrix} 1 \\ x_{1}^{(i)} \\ \cdots \\ x_{n}^{(i)} \end{bmatrix} =  \theta_{0} + \theta_{1}x_{1}^{(i)} + \theta_{2}x_{2}^{(i)} + \cdots + \theta_{n}x_{n}^{(i)} 
$$

where $\theta$ contains the regression model's parameters and $\theta_{0}$ is the bias term.

In order to achieve the hypothesis for all the instances, we use the following array dot product:

$$
h_{\theta}(\mathbf{X}) = \mathbf{X} \theta = \begin{bmatrix} 1 & x_{1}^{(1)} & \cdots & x_{n}^{(1)} \\ 1 & x_{1}^{(2)} & \cdots & x_{n}^{(2)} \\ \vdots & \vdots & \cdots & \vdots \\ 1 & x_{1}^{(m)} & \cdots & x_{n}^{(m)}\\ \end{bmatrix} \begin{bmatrix}\theta_{0} \\ \theta_{1} \\ \cdots \\ \theta_{n}\end{bmatrix} 
$$

We need to train this model in order to find the best parameters. Training a model means setting its parameters so that the model best fits the training set. For this purpose, we first need a measure of how well (or poorly) the model fits the training data. The most common performance measure of a regression model is the Mean Square Error (MSE) which is given below:

$$
MSE (\mathbf{X}, \theta) = \frac{1}{m} \sum_{i=1}^{m} \left(y^{(i)} -  \theta^{T} \mathbf{x}^{(i)}\right)^{2}
$$

In order to achieve the hypothesis for all the instances, we use the following array dot product (let's get rid of $\frac{1}{m}$):

$$
MSE (\mathbf{X}, \theta) = \left(y - \mathbf{X} \theta\right)^{T} \left( y - \mathbf{X} \theta\right)
$$

Taking derivative with respect to $\theta$ and equating to zero, we get,

$$
\hat{\theta}_{OLS} = \left(\mathbf{X}^{T} \cdot \mathbf{X} \right)^{-1} \cdot \mathbf{X}^{T}y
$$

The above is also called the least-squares solution (since we minimized a sum-of-squared-differences objective). 

The matrix $\mathbf{X}^{T} \cdot \mathbf{X}$ is called the Gramian matrix of the design matrix $\mathbf{X}$. It is invertible if and only if the columns of the design matrix are linearly independent, i.e., if and only if the design matrix has full rank. 

Recall from ordinary regression that, we obtain predictions using:

$$
\hat{y} = \mathbf{X} \hat{\theta}_{OLS} = \mathbf{X} \left(\mathbf{X}^{T} \cdot \mathbf{X} \right)^{-1} \cdot \mathbf{X}^{T}y = H y
$$

Here, the matrix $H = \mathbf{X} \left(\mathbf{X}^{T} \cdot \mathbf{X} \right)^{-1} \cdot \mathbf{X}^{T}$ is called the hat matrix or the influence matrix since it transforms $y$ into $\hat{y}$. It is  symmetric, i.e., $H^{T} = H$  and idempotent. In linear algebra, an idempotent matrix is a matrix which, when multiplied by itself, yields itself. That is, $H^{T} H = H^{2} = H$. The trace of an idempotent matrix equals the rank of the matrix.

Residuals are then computed as:

$$
e = y - \hat{y}= y - \mathbf{X} \hat{\theta}_{OLS} = y - \mathbf{X} \left(\mathbf{X}^{T} \cdot \mathbf{X} \right)^{-1} \cdot \mathbf{X}^{T}y = y \left(I - \mathbf{X} \left(\mathbf{X}^{T} \cdot \mathbf{X} \right)^{-1} \cdot \mathbf{X}^{T} \right)
$$

Similarly, the matrix $I - \mathbf{X} \left(\mathbf{X}^{T} \cdot \mathbf{X} \right)^{-1} \cdot \mathbf{X}^{T} = I - H$ is also symmetric, i.e., $(I − H)^{T} = I − H$.  and idempotent. $(I − H)^{2} = (I − H)^{T}(I − H) = I − H − H + H^{2}$. But, since $H$ is idempotent, $H^{2} = H$, and thus $(I − H)^{2} = (I − H)$.

The degrees of freedom of ordinary regression is then $\text{df}_{OLS} = Tr\left(I \right) - Tr\left(\mathbf{X} \left(\mathbf{X}^{T} \cdot \mathbf{X} \right)^{-1} \cdot \mathbf{X}^{T}\right)$ where $Tr$ defines the trace of a matrix. In particular, if $\mathbf{X}$ is of full rank, i.e. $rank(\mathbf{X}) = n $, then, $Tr\left(\mathbf{X} \left(\mathbf{X}^{T} \cdot \mathbf{X} \right)^{-1} \cdot \mathbf{X}^{T}\right)= Tr\left( \mathbf{X}^{T} \cdot \mathbf{X} \left(\mathbf{X}^{T} \cdot \mathbf{X} \right)^{-1} \right)=n$, which is number of predictors in the dataset. $Tr\left(I \right) = m$ because $I$ is the identity matrix where its diagonal elements are 1 and off-diagonal elements are zero. Therefore, the degrees of freedom of ordinary regression is calculated by $m-n$.

## Ridge Regression
Ridge Regression (also known as _Tikhonov Regularization_) is a regularized version of Linear Regression: a regularization term equal to $\lambda \sum_{j=1}^{n} \theta_{j}^{2}$ is added to the cost function. It is nothing but additional burden to error. This forces the learning algorithm to not only fit the data but also keep the model weights as small as possible. Note that the bias term $\theta_{0}$ is not regularized (the sum starts at $i=1$, not $0$). If we define $\theta$ as the vector of feature weights ($\theta_{1}$ to $\theta_{n}$), then the regularization term is simply equal to $(\lVert \theta \rVert_{2})^{2}$, where $\lVert \cdot \rVert_{2}$ represents $L_{2}$ norm of the weight vector. 

Note that regularization parameter needs to be added to the cost function during training. Once the model is trained, you want to evaluate the model's performance using the unregularized performance measure. 

The cost function for ridge regression is expressed as

$$
\begin{split}
MSE (\mathbf{X}, \theta)_{L_{2}} &= \sum_{i=1}^{m} \left(y^{(i)} -  \theta^{T} \mathbf{x}^{(i)}\right)^{2} + \lambda \sum_{j=1}^{n} \theta_{j}^{2}\\
&=\left(y - \mathbf{X} \theta\right)^{T} \left( y - \mathbf{X} \theta\right) + \lambda \lVert \theta \rVert_{2}^{2}\\
&= \left(y - \mathbf{X} \theta\right)^{T} \left( y - \mathbf{X} \theta\right) + \lambda \theta^{T} \theta
\end{split}
$$

Hyperparameter $\lambda > 0$ (the shrinkage parameter) is a complexity parameter that controls the how much you want to regularize the model: the larger the value of $\lambda$, the greater the amount of shrinkage. The coefficients are shrunk toward zero (and each other). If $\lambda = 0$ then Ridge Regression is just Linear Regression. If $\lambda$ is very large, then all the weights end up very close to zero and the results is flat line going through the data's mean. The idea of penalizing by the sum-of-squares of the parameters is also used in neural networks, where it is known as _weight decay_.

The ridge solutions are not equivariant under scaling of the inputs, and so one normally standardizes the inputs before solving the equation above. In addition, do not forget that the intercept $\theta_{0}$ should be left out of the penalty term. We estimate $\theta_{0}$ by $\hat{y} = \frac{1}{m} \sum_{i=1}^{m} y_{i}$. The remaining coefficients get estimated by a ridge regression without intercept, using the centered $x_{j}^{(i)}, \,\,\, i=1,2,...,n\,\, \text{and}\,\, j=1,2,...,m$. Henceforth we assume that this centering has been done, so that the input matrix $\mathbf{X}$ has $n$ (rather than $n + 1$) columns.

This cost function is convex and has a unique solution. One can minimize this criterion using straightforward applications of matrix calculus, as was conducted for the classical OLS criterion for multiple regression. That is, setting to zero and taking the first derivative, we obtain

$$
\begin{split}
\frac{\partial}{\partial \theta}MSE (\mathbf{X}, \theta)_{L_{2}} &= -2 \mathbf{X}^{T} (y-  \mathbf{X}\theta) + 2\lambda \theta = 0\\
&  = -2 \mathbf{X}^{T}y + (2\mathbf{X}^{T}\mathbf{X} + 2\lambda I) \theta = 0
\end{split}
$$

This expression can be further simplified and therefore the ridge estimators are

$$
\hat{\theta}_{ridge} = \left(\mathbf{X}^{T} \cdot \mathbf{X} +\lambda I\right)^{-1} \cdot \mathbf{X}^{T} y
$$

where $I$ is the $n \times n$ identity matrix. We can easily see that ridge estimator is also a linear estimator.

Since we are adding a positive constant to the diagonal of $\mathbf{X}^{T} \cdot \mathbf{X}$ (this is the "ridge" that gives ridge regression its name), we are, in general, producing an invertible matrix, $\mathbf{X}^{T} \cdot \mathbf{X} +\lambda I$, even if $\mathbf{X}^{T} \cdot \mathbf{X}$ is singular. Thus, there is always a unique solution $\hat{\theta}_{ridge}$.

Historically, this particular aspect of ridge regression was the main motivation behind the adoption of this particular extension of OLS theory. In addition, this also shows that $\hat{\theta}_{ridge}$ is still a linear function of the observed values, $y$.

As with Linear Regression, we can perform Ridge Regression either by computing a closed form equation or by performing Gradient Descent. 

Note that the number of degrees of freedom in ridge regression is different than in the regular OLS, since $Tr \left(\mathbf{X} \left(\mathbf{X}^{T} \cdot \mathbf{X} +\lambda I\right)^{-1} \cdot \mathbf{X}^{T} \right)$ is no longer equal to $n$. Usually in a linear-regression fit with $n$ variables, the degrees-of-freedom of the fit is $n$, the number of free parameters. Continuing the analogy of Linear Regression, the degrees of freedom of ridge regression is given by the trace of $\mathbf{X} \left(\mathbf{X}^{T} \cdot \mathbf{X} +\lambda I\right)^{-1} \cdot \mathbf{X}^{T}$ matrix:

$$
\text{df}_{ridge} = Tr \left(\mathbf{X} \left(\mathbf{X}^{T} \cdot \mathbf{X} +\lambda I\right)^{-1} \cdot \mathbf{X}^{T} \right) =\sum_{j=1} ^{n} \frac{d_{j}^{2}}{d_{j}^{2} + \lambda} 
$$

where $d_{j}$ are the eigenvalues of $\mathbf{X}^{T} \cdot \mathbf{X}$. Degrees of freedom is a monotone decreasing function of $\lambda$ with $df_{ridge} = n$ at $\lambda = 0$ and $df_{ridge} = 0$ at $\lambda = \infty$. Of course there is always an additional one degree of freedom for the intercept, which was removed apriori.

Furthermore, the ridge regression estimator is related to the classical OLS estimator and is a biased estimator. Remember that the Gauss-Markov theorem states that the OLS estimate for regression coefficients is the BLUE, so by using ridge regression, we are sacrificing some benefits of OLS estimators in order to constrain estimator variance. Estimators constructed using ridge regression are in fact biased:

$$
\begin{split}
\hat{\theta}_{ridge} &= \left(\mathbf{X}^{T} \cdot \mathbf{X} +\lambda I\right)^{-1} \cdot \left(\mathbf{X}^{T} \cdot \mathbf{X} \right) \hat{\theta}_{OLS}\\
&= \left[I + \lambda \left(\mathbf{X}^{T} \cdot \mathbf{X}\right)^{-1}\right]^{-1} \hat{\theta}_{OLS}
\end{split}
$$

Let’s compute the mean and variance:

$$
\begin{split}
E(\hat{\theta}_{ridge}) &= E\left[\left(\mathbf{X}^{T} \cdot \mathbf{X} +\lambda I\right)^{-1} \cdot \mathbf{X}^{T} y\right]\\
&= \left(\mathbf{X}^{T} \cdot \mathbf{X} +\lambda I\right)^{-1} \cdot \mathbf{X}^{T} E(y)\\
&= \left(\mathbf{X}^{T} \cdot \mathbf{X} +\lambda I\right)^{-1} \cdot \mathbf{X}^{T} \mathbf{X}\theta
\end{split}
$$

and

$$
\begin{split}
Var(\hat{\theta}_{ridge}) &= Var\left[\left(\mathbf{X}^{T} \cdot \mathbf{X} +\lambda I\right)^{-1} \cdot \mathbf{X}^{T} y\right]\\
&= \left(\mathbf{X}^{T} \cdot \mathbf{X} +\lambda I\right)^{-1} \cdot \mathbf{X}^{T} Var(y) \mathbf{X}\left(\mathbf{X}^{T} \cdot \mathbf{X} +\lambda I\right)^{-1}\\
&=  \left(\mathbf{X}^{T} \cdot \mathbf{X} +\lambda I\right)^{-1} \cdot \mathbf{X}^{T} \sigma^{2} \mathbf{X} \mathbf{I}\left(\mathbf{X}^{T} \cdot \mathbf{X} +\lambda I\right)^{-1}\\
&= \sigma^{2} \left(\mathbf{X}^{T} \cdot \mathbf{X} +\lambda I\right)^{-1} \cdot \mathbf{X}^{T}\mathbf{X} \mathbf{I}\left(\mathbf{X}^{T} \cdot \mathbf{X} +\lambda I\right)^{-1}
\end{split}
$$

Notice how both of these expressions smoothly approach the corresponding formulas ones for ordinary least squares as $\lambda \to 0$. Indeed, under the Gaussian noise assumption, $\hat{\theta}_{ridge} $ actually has a Gaussian distribution with the given expectation and variance.

Additionally, when $X$ is composed of orthonormal variables, such that $\mathbf{X}^{T} \cdot \mathbf{X} = I_{n}$, it then follows that

$$
\hat{\theta}_{ridge} = (I + \lambda I)^{-1}\cdot \mathbf{X}^{T} y = ((1+\lambda)I)^{-1}\cdot \mathbf{X}^{T} y = \frac{1}{1+\lambda}\hat{\theta}_{OLS}
$$

This is instructive, as it shows that, in this simple case, the ridge estimator is simply a scaled version of the OLS estimator.

**NOTE**: It is important to scale the data before performing Ridge Regression, as it is sensitive to the scale of the input features. In practice we center and scale the covariates. This holds true for the most regularized models. 

## Ridge Regression in Scikit-Learn

Here is how to perform Ridge Regression with Scikit-Learn using a closed-form solution  (a variant of $\hat{\theta}_{ridge}$ equation above using a matrix factorization technique by Andre-Louis Cholesky).

{% highlight python %}
#Let's create a data with 20 observations and 1 variable.
import numpy as np

np.random.seed(42)
m = 20
X = 3 * np.random.rand(m, 1)
y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5

from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])
#array([[1.55071465]])
{% endhighlight %}

and using Stochastic Gradient Descent

{% highlight python %}
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=50, tol=-np.infty, penalty="l2", random_state=42)
sgd_reg.fit(X, y.ravel())
sgd_reg.predict([[1.5]])
#array([1.49905184])
{% endhighlight %}

Alternatively you can use the `Ridge` class with the "sag" solver. Stochastic Average GD is a variant of Stochastic Gradient Descent.

{% highlight python %}
ridge_reg = Ridge(alpha=1, solver="sag", random_state=42)
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])
#array([[1.5507201]])
{% endhighlight %}

# Lasso Regression
We have seen that ridge regression essentially re-scales the OLS estimates. The Least Absolute Shrinkage and Selection Operator Regression (simply called Lasso Regression), by contrast, tries to produce a sparse solution, in the sense that several of the slope parameters will be set to zero. This means that the lasso is somewhere between subset selection and ridge regression. 

One may therefore refers to ridge regression as soft thresholding, whereas the lasso is soft/hard, and subset selection is a hard thresholding; since, in the latter, only a subset of the variables are included in the final model. Lasso uses $L_{1}$ norm of the weight vector instead of the square of $L_{2}$ norm (the squared length of the coefficient vector). It involves penalizing the absolute size of the regression coefficients. Again, we have a tuning parameter $\lambda$ that controls the amount of regularization (we can also think of $\lambda$ in Lasso Regression as a parameter controlling the level of sparsity).

LASSO coefficients are the solutions to the $L_{1}$ optimization problem:

$$
\text{minimize}\,\,\,\,\, \left(y - \mathbf{X} \theta\right)^{T} \left( y - \mathbf{X} \theta\right) \,\,\,\,\, \text{s.t.} \sum_{j=1}^{n} |\theta_{j}| \leq t \,\,\,\,\,\text{for some value of t}.
$$

there exists a one-to-one correspondence between $t$ and $\lambda$. To solve the constrained optimization, Lagrange multipliers can be employed. Therefore, this is equivalent to loss function :

$$
\begin{split}
MSE (\mathbf{X}, \theta)_{L_{1}} &= \sum_{i=1}^{m} \left(y^{(i)} -  \theta^{T} \mathbf{x}^{(i)}\right)^{2} + \lambda \sum_{j=1}^{n} |\theta_{j}|\\
&= \left(y - \mathbf{X} \theta\right)^{T} \left( y - \mathbf{X} \theta\right) + \lambda \lVert \theta \rVert_{1}
\end{split}
$$

This has the nice property that it tends to give sparse solutions -it sets coefficients to be exactly zero (unlike ridge), so we can say that Lasso automatically does feature selection. As $\lambda$ increases, more and more coefficients are set to zero and eliminated (theoretically, when $\lambda = \infty$, all coefficients are eliminated).

Contrary to ridge regression, $\hat{\theta}_{lasso}$ has no closed-form solution (with a single predictor, the LASSO solution has a closed form solution). The L1-penalty makes the solution non-linear in the $y$'s. The above constrained minimization is a quadratic programming problem, whose solution can be efficiently approximated. Coordinate Descent can be used  to solve this problem. 

## Lasso Regression in Scikit-Learn

{% highlight python %}
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]])
#array([1.53788174])
{% endhighlight %}

Note that you could also use an `SGDRegressor(penalty="l1")`

**NOTE**: L1 corresponds to setting a Laplacean prior on the terms, while L2 corresponds to a Gaussian prior.

# Elastic Net

Use of Lasso penalty function has several limitations. For example, in the "large number of predictors $p$, small number of observation $n$" case (high-dimensional data with few examples), the LASSO selects at most $n$ variables before it saturates. Also if there is a group of highly correlated variables, then the LASSO tends to select one variable from a group and ignore the others. To overcome these limitations, the elastic net adds a quadratic part to the penalty, which is, when used alone is ridge regression (known also as Tikhonov regularization).

Elastic Net is a middle ground between Ridge Regression and Lasso Regression. The regularization term is a simple mix of both Ridge and Lasso's regularization terms and you can control the mix ratio $r$. When $r=0$ Elastic Net is equivalent to Ridge Regression and when $r=1$, it is equivalent to Lasso Regression. Therefore we can choose an $r$ value between 0 and 1 to optimize the elastic net. Effectively this will shrink some coefficients and set some to 0 for sparse selection.

Elastic Net aims at minimizing the following loss function:

$$
\begin{split}
MSE (\mathbf{X}, \theta)_{\text{Elastic Net}} &= \sum_{i=1}^{m} \left(y^{(i)}  - \theta^{T} \mathbf{x}^{(i)}\right)^{2} + \lambda \left( r \sum_{j=1}^{n} |\theta_{j}| + (1-r) \sum_{j=1}^{n} \theta_{j}^{2} \right)\\
& = \left(y - \mathbf{X} \theta\right)^{T} \left( y - \mathbf{X} \theta\right) + \lambda \left( r \lVert \theta \rVert_{1} + (1-r) \lVert \theta \rVert_{2}^{2}\right)
\end{split}
$$

## Elastic Net in Scikit-Learn

{% highlight python %}
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])
#array([1.54333232])
{% endhighlight %}

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/sphx_glr_plot_sgd_penalties_001.png?raw=true)

# Some Notes and Summary
1. Complex models have less bias; simpler models have less variance. Regularization encourages simpler models.  In essence, $\lambda$ controls the model complexity. Generally speaking, the bias increases and the variance decreases as $\lambda$ (amount of shrinkage) increases, which will lead to underfitting due to the simple model we have and vice versa. Everything that reduces the dimensionality of a problem is biased. With less variables there are less degrees of freedom but also less "chances" to represent the dependent (response) variable. 

2. The complexity parameter can be chosen with cross validation or using some model selection criterion such as AIC. 

3. $L_{1}$ penalizes sum of absolute value of weights. $L_{1}$ has a sparse solution. $L_{1}$ has multiple solutions. $L_{1}$ has built in feature selection. $L_{1}$ is robust to outliers. $L_{1}$ generates model that are simple and interpretable but cannot learn complex patterns.

4. $L_{2}$ regularization penalizes sum of square weights. $L_{2}$ has a non sparse solution. $L_{2}$ has one solution. $L_{2}$ has no feature selection. $L_{2}$ is not robust to outliers. $L_{2}$ gives better prediction when output variable is a function of all input features. $L_{2}$ regularization is able to learn complex data patterns.

# $L_{1}$ and $L_{2}$ as loss function

$L_{1}$ Loss Function is used to minimize the error which is the sum of the all the absolute differences between the true value and the predicted value.

$$
\text{$L_{1}$ Loss} = \sum_{i=1}^{n} \lvert y_{true} - y_{predicted} \rvert
$$

$L_{2}$ Loss Function is used to minimize the error which is the sum of the all the squared differences between the true value and the predicted value.

$$
\text{$L_{2}$ Loss} = \sum_{i=1}^{n} (y_{true} - y_{predicted})^{2}
$$

If the difference between actual value and predicted value is large, the squared difference would be larger. So, squared error approach penalizes large errors more as compared to absolute error approach. Therefore, if you want the model to penalize large errors more, minimizing squared error would be better.

Among the main differences between the two are that using the squared error is easier to solve for and using the absolute error is more robust to outliers.

When the outliers are present in the dataset, then the L2 Loss Function does not perform well. The reason behind this bad performance is that if the dataset is having outliers, then because of the consideration of the squared differences, it leads to the much larger error. Hence, $L_{2}$ Loss Function is not useful here. Prefer $L_{1}$ Loss Function as it is not affected by the outliers or remove the outliers and then use $L_{2}$ Loss Function.

The reason that the squared error is easier to solve for is that the derivatives are continuous. In the case of linear regression, this means that you can find the solution in closed form (by setting the derivative to zero). Linear regression with absolute error requires an iterative approach, which is more complicated and isn't as efficient. For other models, even if the solution can't be found in closed form, it's usually easier to solve for because simple methods such as gradient descent (and related methods) can be used.

# Why does $L_{1}$ regularization cause parameter sparsity where $L_{2}$ regularization does not?

For simplicity, let's consider two-dimensional feature vectors and mean squared error:

$$
h_{\theta_{0}, \theta_{1}, \theta_{2}}(\mathbf{x}^{(i)}) =  \theta_{0} + \theta_{1}x_{1}^{(i)} + \theta_{2}x_{2}^{(i)}
$$

The optimization problem we have here is

$$
\begin{split}
\min_{\theta_{0}, \theta_{1}, \theta_{2}}  MSE (\mathbf{X}, \theta_{0}, \theta_{1}, \theta_{2}) &= \frac{1}{m} \sum_{i=1}^{m} \left(h_{\theta_{0}, \theta_{1}, \theta_{2}}(\mathbf{x}^{(i)}) - y^{(i)}\right)^{2}\\
&= \frac{1}{m} \sum_{i=1}^{m} \left( \theta_{0} + \theta_{1}x_{1}^{(i)} + \theta_{2}x_{2}^{(i)} - y^{(i)}\right)^{2}\\
\end{split}
$$

With $L_{1}$ regularization, the optimization problem becomes:

$$
\min_{\theta_{0}, \theta_{1}, \theta_{2}} \left[\lambda \left( \mid \theta_{1} \mid + \mid \theta_{2} \mid \right) + \frac{1}{m} \sum_{i=1}^{m} \left( \theta_{0} + \theta_{1}x_{1}^{(i)} + \theta_{2}x_{2}^{(i)} - y^{(i)}\right)^{2}\right]
$$

With $L_{2}$ regularization

$$
\min_{\theta_{0}, \theta_{1}, \theta_{2}} \left[ \lambda \left((\theta_{1})^{2} + (\theta_{2})^{2} \right) + \frac{1}{m} \sum_{i=1}^{m} \left( \theta_{0} + \theta_{1}x_{1}^{(i)} + \theta_{2}x_{2}^{(i)} - y^{(i)}\right)^{2}\right]
$$

By adding the size of the weights to the loss function, we force the minimization algorithm to seek for such solution that along with minimizing the loss function, would make the "size" of weights smaller. 

$\lambda$ plays the role of a trade-off between model complexity and underfitting. Once this optimal value for $\lambda$ is found by cross-validation or validation set, we can say that we have found a certain optimal value for regularization term.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Screen%20Shot%202019-10-13%20at%2014.36.11.png?raw=true)

By looking at Figure, you can see why $L_{1}$ acts as a feature selector. The optimum of $L_{1}$-regularized cost is always found on one or more axes, which means that one or more $\theta_{j}$ in the solution is zero. So, the corresponding feature is excluded from the model.

In the same Figure, the contours of the elipses correspond to sub-optimal solutions for unregularized cost function. All solutions on the contour of some ellipse have the same value. Smallar the ellipse, the closer the solution of this optimization problem to the optimum. In other words, as we move away from the center of the ellipse, loss increases quadratically.  The other term, which belongs to regularization, is minimized at the origin, which all parameters equal to zero. It increases, again, quadratically away. Therefore, the combination of data plus regularization, will be minimized at some point that touches both surfaces, since in order to reduces the regularization we would have to leave this isosurface and increase the data loss and vice-versa. We cannot reduce either loss without increasing the other.

You can make sure that indeed the equation $\frac{1}{m} \sum_{i=1}^{m} \left( \theta_{0} + \theta_{1}x_{1}^{(i)} + \theta_{2}x_{2}^{(i)} - y^{(i)}\right)^{2} = r$ for some $r$ is an ellipse by creating two or more random training examples in the form $(\mathbf{x}, y)$, fixing $\theta_{0}$ and $r$ at some values. $\theta_{0}$ only affects the position of the ellipse not its form while $r$ affect the only its size. Once you fixed all values the only remaning variables are  $\theta_{1}$ and  $\theta_{2}$.

Notice that you need at least two different training examples because with one example, you will obtain two parallel lines not ellipse.

# A Probabilistic Interpretation of Regularization

In this section, we will look at a probabilistic (Bayesian) interpretation of regularization. We'll take a look at both L1 and L2 regularization in the context of ordinary linear regression. Remember that in a Bayesian analysis we multiply the likelihood function by a prior probability for the parameter in order to obtain the posterior distribution.

In the context of a ordinary linear regression, we have MSE that we try to minimize:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Page0001-1.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Page0002-1.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Page0003-1.png?raw=true)
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/Page0004-1.png?raw=true)

# REFERENCES

1. [https://web.stanford.edu/~hastie/Papers/ESLII.pdf](https://web.stanford.edu/~hastie/Papers/ESLII.pdf){:target="_blank"}
2. [https://www.ias.ac.in/article/fulltext/reso/023/04/0439-0464](https://www.ias.ac.in/article/fulltext/reso/023/04/0439-0464){:target="_blank"}

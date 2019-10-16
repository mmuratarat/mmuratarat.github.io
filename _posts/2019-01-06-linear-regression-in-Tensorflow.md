---
layout: post
title: "Linear Regression in Tensorflow"
author: "MMA"
comments: true
---
Linear regression (LR) is a linear approach to modelling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables). More generally, a linear model makes a prediction by simply computing a weighted sum of the input features, plus a constant the *bias term* (also called the *intercept*) as shown below:

$$\hat{y} = \theta_{0} + \theta_{1}x_{1} + \theta_{2}x_{2} + \cdots + \theta_{n}x_{n} $$

where $\hat{y}$ is the predicted values, $n$ is the number of features in the data, $x_{i}$ is the $i$th feature value and $\theta_{j}$ is the $j$th model parameter (including the bias term $\theta_{0}$).

This equation above can be written much more concisely using a vectorized form (here, it is [dot product](http://wiki.fast.ai/index.php/Linear_Algebra_for_Deep_Learning)):

$$\hat{y} = h_{\theta} ( \mathbf{x} ) =  \theta^{T} \mathbf{x}$$

where $\theta$ is the model's parameter vector, containing the bias term $\theta_{0}$ and the feature weights, i.e., $\theta_{1}$ to $\theta_{n}$. $\theta^{T}$ is the transpose of $\theta$ (a row vector instead of a column vector). $\mathbf{x}$ is the instance's feature vector, containing $x_{0}$ to $x_{n}$ with $x_{0}$ always equal to $1$. $\left(\theta^{T} \cdot  \mathbf{x} \right)$ is the dot product of $\theta^{T}$ and $\mathbf{x}$. $h_{\theta} (\cdot)$ is the hypothesis function, using the model parameters $\theta$.

We need to train this model in order to find the best parameters. Training a model means setting its parameters so that the model best fits the training set. For this purpose, we first need a measure of how well (or poorly) the model fits the training data. The most common performance measure of a regression model is the Mean Square Error (MSE) which is given below:

$$MSE (\mathbf{X}, \theta) = \frac{1}{m} \sum_{i=1}^{m} \left(\theta^{T} \mathbf{x}^{(i)} - y^{(i)}\right)^{2}$$

We'll define the "design matrix" $\mathbf{X}$ as a matrix of $m$ rows (observations), in which each row is the $i$-th sample (the vector $\mathbf{x}^{(i)}$). With this, we can rewrite the MSE cost as following, replacing the explicit sum by matrix multiplication:

$$MSE (\mathbf{X}, \theta) = \frac{1}{m} \left( \mathbf{X} \theta - \mathbf{y}\right)^{T} \left( \mathbf{X} \theta - \mathbf{y}\right)$$

Therefore, to train a LR model, you need to find the value of $\theta$ that minimizes the MSE.

In order to find the value of $\theta$ that minimized the cost function, there is a closed-form solution - in other words, a mathematical equation that gives the results directly. This is called the *Normal Equations*:

$$\hat{\theta} = \left(\mathbf{X}^{T} \cdot \mathbf{X} \right)^{-1} \cdot \mathbf{X}^{T} \cdot \mathbf{y} $$

where $\hat{\theta}$ is the estimation of $\theta$ that minimizes the cost function and $\mathbf{y}$ is the vector of target values containing $y^{(1)}$ to $y^{(m)}$.

Other approach that you can take to compute $\hat{\theta}$ is to use Gradient Descent algorithm. To implement it, you need to compute the gradient of the cost function with regards to each model parameter $\theta_{j}$. In other words, you need to calculate how much the cost function will change if you change $\theta_{j}$ just a little. This is called a *partial derivative*.

Partial derivatives of this cost function with regards to $\theta_{j}$ are computed as follows:

$$\dfrac{\partial}{\partial \theta_{j}} MSE (\theta) = \frac{2}{m} \sum_{i=1}^{m} \left(\theta^{T} \mathbf{x}^{(i)} - y^{(i)}\right) x_{j}^{(i)}$$

Instead of computing these partial derivatives individually, you can use the equation below to compute all in one go. The gradient vector, noted as $\nabla_{\theta} \text{MSE} (\theta)$, contains all the partial derivatives of the cost function (one for each model parameters).

$$\nabla_{\theta} \text{MSE} (\theta) = \begin{bmatrix}\dfrac{\partial}{\partial \theta_{0}} MSE (\theta)\\ \dfrac{\partial}{\partial \theta_{1}} MSE (\theta) \\ \vdots  \\ \dfrac{\partial}{\partial \theta_{n}} MSE (\theta)\end{bmatrix} = \dfrac{2}{m} \mathbf{X}^{T} \cdot \left(\mathbf{X} \cdot \theta - \mathbf{y} \right) $$

Note that if we set this derivative to zero, we obtain normal equations $\left(\mathbf{X}^{T}\mathbf{X}\right)^{-1} \mathbf{X}\mathbf{y}$.

Once you have the gradient vector, which points uphill, just go in the opposite direction to go downhill. This means subtracting $\nabla_{\theta} \text{MSE} (\theta)$ from $\theta$. Because, basically, MSE cost function happens to be a convex optimization problem (because MSE equation is a convex function. Its Hessian is positive semi-definite. Look [here](https://math.stackexchange.com/a/2774380/45210){:target="_blank"} for proof) and we are trying find one global minimum of it. 

In order to get next step of $\theta$, you use the formula below:

$$\theta^{(\text{next step})} = \theta - \alpha \nabla_{\theta} \text{MSE} (\theta)$$

where, here, $\alpha$ is the learning rate, a.k.a., the size of steps in Gradient Descent.

# Implementing Linear Regression in Tensorflow
## Implementing Normal Equations in Tensorflow

<script src="https://gist.github.com/mmuratarat/2aa8efb88ad96be19791ad15910beef2.js"></script>

Let's compare it with `sklearn` module.

<script src="https://gist.github.com/mmuratarat/46012e47764178d5d746a3ae2cdd70fa.js"></script>

## Implementing Gradient Descent in Tensorflow

### Manually Computing the Gradients

<script src="https://gist.github.com/mmuratarat/dbb7caac0f55339cea93f1f5b8ba91f4.js"></script>

### Using autodiff

<script src="https://gist.github.com/mmuratarat/39f9b94ed6a84a800ad60c0f862808fe.js"></script>

### Using an Optimizer

<script src="https://gist.github.com/mmuratarat/0a1045a14212c3a2f9e9e10ff54a0889.js"></script>


**NOTE**: Do not forget to reset graph and set seed if you want to have a reproducible results.
<script src="https://gist.github.com/mmuratarat/c6d227805e351010c0dbfcd0353e8439.js"></script>

**NOTE:** In some books/tutorials/articles, you can see that the cost function for linear regression is defined as follows:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left(\theta^{T} \mathbf{x}^{(i)} - y^{(i)}\right)^{2}$$

The $\frac{1}{m}$ is to "average" the squared error over the number of observations so that the number of observations doesn't affect the function.

So now the question is why there is an extra $\frac{1}{2}$. In short, it is merely for convenience, and actually, so is the $m$ - they're both constants. The expression for the gradient becomes prettier with the $\frac{1}{2}$, because the $2$ from the square term cancels out. However, the $m$ is useful if you solve this problem with gradient descent because it will take out the dependency on the number of observations. Then your gradient becomes the average of $m$ terms instead of a sum, so it's scale does not change when you add more data points.

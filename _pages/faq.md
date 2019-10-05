---
layout: post
title: "Frequently Asked Questions (and Answers)"
author: MMA
social: true
comments: false
permalink: /faq/
---
[Linear Algebra](#linear-algebra)

1. [What are scalars, vectors, matrices, and tensors?](#what-are-scalars-vectors-matrices-and-tensors)
2. [How do you normalize a vector?](#how-do-you-normalize-a-vector)
2. [What is Hadamard product of two matrices?](#what-is-hadamard-product-of-two-matrices)
3. [What is a scalar valued function?](#what-is-a-scalar-valued-function)
4. [What is a vector valued function?](#what-is-a-vector-valued-function)
5. [What is the gradient?](#what-is-the-gradient)
6. [What is a Jacobian matrix?](#what-is-a-jacobian-matrix)
7. [What is a Hessian matrix?](#what-is-a-hessian-matrix)
8. [What is an identity matrix?](#what-is-an-identity-matrix)
9. [What is the transpose of a matrix?](#what-is-the-transpose-of-a-matrix)
10. [What is an inverse matrix?](#what-is-an-inverse-matrix)
11. [When does inverse of a matrix exist?](#when-does-inverse-of-a-matrix-exist)
12. [If inverse of a matrix exists, how to calculate it?](#if-inverse-of-a-matrix-exists-how-to-calculate-it)
13. [What is the determinant of a square matrix? How is it calculated? What is the connection of determinant to eigenvalues?](#what-is-the-determinant-of-a-square-matrix-how-is-it-calculated-what-is-the-connection-of-determinant-to-eigenvalues)
14. Discuss span and linear dependence.
15. What is Ax = b? When does Ax =b has a unique solution?
16. In Ax = b, what happens when A is fat or tall?
17. [What is a norm? What is $L^{1}$, $L^{2}$ and $L^{\infty}$ norm? What are the conditions a norm has to satisfy?](#what-is-a-norm-what-is-l_1-l_2-and-l_infty-norm-what-are-the-conditions-a-norm-has-to-satisfy)
18. Why is squared of L2 norm preferred in ML than just L2 norm?
19. When L1 norm is preferred over L2 norm?
20. Can the number of nonzero elements in a vector be defined as $L^{0}$ norm? If no, why?
21. [What is Frobenius norm?](#what-is-frobenius-norm)
22. [What is a diagonal matrix?](#what-is-a-diagonal-matrix)
23. Why is multiplication by diagonal matrix computationally cheap? How is the multiplication different for square vs. non-square diagonal matrix?
24. At what conditions does the inverse of a diagonal matrix exist?
25. [What is a symmetric matrix?](#what-is-a-symmetric-matrix)
26. What is a unit vector?
27. When are two vectors x and y orthogonal?
28. At $\mathbb{R}^n$ what is the maximum possible number of orthogonal vectors with non-zero norm?
29. When are two vectors x and y orthonormal?
30. What is an orthogonal matrix? Why is computationally preferred?
31. [What is eigendecomposition, eigenvectors and eigenvalues? How to find eigenvalues of a matrix?](#what-is-eigendecomposition-eigenvectors-and-eigenvalues-how-to-find-eigenvalues-of-a-matrix)
36. What is Singular Value Decomposition? Why do we use it? Why not just use ED?
37. Given a matrix A, how will you calculate its Singular Value Decomposition?
38. What are singular values, left singulars and right singulars?
39. What is the connection of Singular Value Decomposition of A with functions of A?
40. Why are singular values always non-negative?
41. What is the Moore Penrose pseudo inverse and how to calculate it?
42. If we do Moore Penrose pseudo inverse on Ax = b, what solution is provided is A is fat? Moreover, what solution is provided if A is tall?
43. Which matrices can be decomposed by ED?
44. Which matrices can be decomposed by SVD?
45. [What is the trace of a matrix?](#what-is-the-trace-of-a-matrix)
46. [How to write Frobenius norm of a matrix A in terms of trace?](#how-to-write-frobenius-norm-of-a-matrix-a-in-terms-of-trace)
47. Why is trace of a multiplication of matrices invariant to cyclic permutations?
48. [What is the trace of a scalar?](#what-is-the-trace-of-a-scalar)
49. [What is Spectral Decomposition?](#what-is-spectral-decomposition)
49. [What do positive definite, positive semi-definite and negative definite/negative semi-definite mean?](#what-do-positive-definite-positive-semi-definite-and-negative-definitenegative-semi-definite-mean)
50. [How to make a positive definite matrix with a matrix that’s not symmetric?](#how-to-make-a-positive-definite-matrix-with-a-matrix-thats-not-symmetric)


[Numerical Optimization](#numerical-optimization)

1. What is underflow and overflow?
2. How to tackle the problem of underflow or overflow for softmax function or log softmax function?
3. What is poor conditioning?
4. What is the condition number?
5. What are grad, div and curl?
6. What are critical or stationary points in multi-dimensions?
7. Why should you do gradient descent when you want to minimize a function?
8. What is line search?
9. What is hill climbing?
10. What is curvature?
11. [Describe convex function.](#describe-convex-function)


[Set Theory](#set-theory)

1. [What is a random experiment?](#what-is-a-random-experiment)
2. [What is a sample space?](#what-is-a-sample-space)
3. [What is an empty set?](#what-is-an-empty-set)
4. [What is an event?](#what-is-an-event)
5. [What are the operations on a set?](#what-are-the-operations-on-a-set)
6. [What is mutually exclusive (disjoint) events?](#what-is-mutually-exclusive-disjoint-events)
7. [What is a non-disjoint event?](#what-is-a-non-disjoint-event)
8. [What is an independent event?](#what-is-an-independent-event)
9. [What is exhaustive events?](#what-is-exhaustive-events)
10. [What is Inclusion-Exlusive Principle?](#what-is-inclusion-exlusive-principle)


[Probability](#probability)

9. [What is a probability?](#what-is-a-probability)
10. [What are the probability axioms?](#what-are-the-probability-axioms)
11. [What is a random variable?](#what-is-a-random-variable)
12. Compare "Frequentist probability" vs. "Bayesian probability"?
13. [What is a probability distribution?](#what-is-a-probability-distribution)
17. [What is a probability mass function? What are the conditions for a function to be a probability mass function?](#what-is-a-probability-mass-function-what-are-the-conditions-for-a-function-to-be-a-probability-mass-function)
18. [What is a probability density function? What are the conditions for a function to be a probability density function?](#what-is-a-probability-density-function-what-are-the-conditions-for-a-function-to-be-a-probability-density-function)
16. [What is a joint probability distribution? What is a marginal probability? Given the joint probability function, how will you calculate it?](#what-is-a-joint-probability-distribution-what-is-a-marginal-probability-given-the-joint-probability-function-how-will-you-calculate-it)
20. [What is conditional probability? Given the joint probability function, how will you calculate it?](#what-is-conditional-probability-given-the-joint-probability-function-how-will-you-calculate-it)
21. [State the Chain rule of conditional probabilities.](#state-the-chain-rule-of-conditional-probabilities)
22. What are the conditions for independence and conditional independence of two random variables?
23. [What are expectation, variance and covariance?](#what-are-expectation-variance-and-covariance)
24. [What is the covariance for a vector of random variables?](#what-is-the-covariance-for-a-vector-of-random-variables)
25. [What is the correlation for a vector of random variables? How is it related to covariance matrix?](#what-is-the-correlation-for-a-vector-of-random-variables-how-is-it-related-to-covariance-matrix)
26. [What is Cross-covariance?](#what-is-cross-covariance)
25. What is moment generating function? What is characteristic function? How to compute them?
26. [What are the properties of Distributions?](#what-are-the-properties-of-distributions)
27. [What are the measures of Central Tendency: Mean, Median, and Mode?](#what-are-the-measures-of-central-tendency-mean-median-and-mode)
28. [What are the properties of an estimator?](#what-are-the-properties-of-an-estimator)
25. [What is a Bernoulli distribution? Calculate the expectation and variance of a random variable that follows Bernoulli distribution?](#what-is-a-bernoulli-distribution-calculate-the-expectation-and-variance-of-a-random-variable-that-follows-bernoulli-distribution)
26. [What is Binomial distribution?](#what-is-binomial-distribution)
26. [What is a multinoulli distribution?](#what-is-a-multinoulli-distribution)
26. [What is a multinomial distribution?](#what-is-a-multinomial-distribution)
27. What is a normal distribution?
28. Why is the normal distribution a default choice for a prior over a set of real numbers?
29. [What is the central limit theorem?](#what-is-the-central-limit-theorem)
30. [What is the sampling distribution of sample proportion, p-hat?](#what-is-the-sampling-distribution-of-sample-proportion-p-hat)
30. What are exponential and Laplace distribution?
31. What are Dirac distribution and Empirical distribution?
32. What is mixture of distributions?
33. Name two common examples of mixture of distributions? (Empirical and Gaussian Mixture)
34. Is Gaussian mixture model a universal approximator of densities?
35. Write the formulae for logistic and softplus function.
36. Write the formulae for Bayes rule.
37. What do you mean by measure zero and almost everywhere?
38. If two random variables are related in a deterministic way, how are the PDFs related?
39. Define self-information. What are its units?
44. What are structured probabilistic models or graphical models?
45. In the context of structured probabilistic models, what are directed and undirected models? How are they represented? What are cliques in undirected structured probabilistic models?
46. [What is population mean and sample mean?](#what-is-population-mean-and-sample-mean)
47. [What is population standard deviation and sample standard deviation?](#what-is-population-standard-deviation-and-sample-standard-deviation)
48. [Why population standard deviation has N degrees of freedom while sample standard deviation has N-1 degrees of freedom? In other words, why 1/N inside root for population and 1/(N-1) inside root for sample standard deviation?](#why-population-standard-deviation-has-n-degrees-of-freedom-while-sample-standard-deviation-has-n-1-degrees-of-freedom-in-other-words-why-1n-inside-root-for-population-and-1n-1-inside-root-for-sample-standard-deviation)
49. [What is the distribution of the sample mean?](#what-is-the-distribution-of-the-sample-mean)
50. [What is the sampling distribution of the sample variance?](#what-is-the-sampling-distribution-of-the-sample-variance)
50. [What is standard error of the estimate?](#what-is-the-standard-error-of-the-estimate)
51. [What is confidence interval?](#what-is-confidence-interval)
52. [What is a p-value?](#what-is-a-p-value)
53. [What do Type I and Type II errors mean?](#what-do-type-i-and-type-ii-errors-mean)


[General Machine Learning](#general-machine-learning)

1. [What is an epoch, a batch and an iteration?](#what-is-an-epoch-a-batch-and-an-iteration)
2. [What is the matrix used to evaluate the predictive model? How do you evaluate the performance of a regression prediction model vs a classification prediction model?](#what-is-the-matrix-used-to-evaluate-the-predictive-model-how-do-you-evaluate-the-performance-of-a-regression-prediction-model-vs-a-classification-prediction-model)
3. [What are the assumptions required for linear regression?](#what-are-the-assumptions-required-for-linear-regression)
4. [What are the assumptions required for logistic regression?](#what-are-the-assumptions-required-for-logistic-regression)
5. [Why is logistic regression considered to be linear model?](#why-is-logistic-regression-considered-to-be-linear-model)
5. [Why sigmoid function in Logistic Regression?](#why-sigmoid-function-in-logistic-regression)
5. [What is Softmax regression and how is it related to Logistic regression?](#what-is-softmax-regression-and-how-is-it-related-to-logistic-regression)
5. [What is collinearity and what to do with it? How to remove multicollinearity?](#what-is-collinearity-and-what-to-do-with-it-how-to-remove-multicollinearity)
6. [What is R squared?](#what-is-r-squared)
7. [You have built a multiple regression model. Your model $R^{2}$ isn't as good as you wanted. For improvement, your remove the intercept term, your model $R^{2}$ becomes 0.8 from 0.3. Is it possible? How?](#you-have-built-a-multiple-regression-model-your-model-r2-isnt-as-good-as-you-wanted-for-improvement-your-remove-the-intercept-term-your-model-r2-becomes-08-from-03-is-it-possible-how)
8. [How do you validate a machine learning model?](#how-do-you-validate-a-machine-learning-model)
9. [What is the Bias-variance trade-off for Leave-one-out and k-fold cross validation?](#what-is-the-bias-variance-trade-off-for-leave-one-out-and-k-fold-cross-validation)
10. [Describe Machine Learning, Deep Learning, Supervised Learning, Unsupervised Learning, Semi-Supervised Learning, Reinforcement Learning with examples](#describe-machine-learning-deep-learning-supervised-learning-unsupervised-learning-semi-supervised-learning-reinforcement-learning-with-examples)
11. [What is batch learning and online learning?](#what-is-batch-learning-and-online-learning)
12. [What is instance-based and model-based learning?](#what-is-instance-based-and-model-based-learning)
13. [What are the main challenges of machine learning algorithms?](#what-are-the-main-challenges-of-machine-learning-algorithms)
14. [What are the most important unsupervised learning algorithms?](#what-are-the-most-important-unsupervised-learning-algorithms)
15. [What is Tensorflow?](#what-is-tensorflow)
16. [Why Deep Learning is important?](#why-deep-learning-is-important)
17. [What are the three respects of an learning algorithm to be efficient?](#what-are-the-three-respects-of-an-learning-algorithm-to-be-efficient)
18. [What are the differences between a parameter and a hyperparameter?](#what-are-the-differences-between-a-parameter-and-a-hyperparameter)
19. [Why do we have three sets: training, validation and test?](#why-do-we-have-three-sets-training-validation-and-test)
20. [What are the goals to build a learning machine?](#what-are-the-goals-to-build-a-learning-machine)
21. [What are the solutions of overfitting?](#what-are-the-solutions-of-overfitting)
22. [Is it better to design robust or accurate algorithms?](#is-it-better-to-design-robust-or-accurate-algorithms)
23. [What is feature engineering?](#what-is-feature-engineering)
23. [What are some feature scaling (a.k.a data normalization) techniques? When should you scale your data? Why?](#what-are-some-feature-scaling-aka-data-normalization-techniques-when-should-you-scale-your-data-why)
24. [What are the types of feature selection methods?](#what-are-the-types-of-feature-selection-methods)
25. [How can you prove that one improvement you've brought to an algorithm is really an improvement over not doing anything?](#how-can-you-prove-that-one-improvement-youve-brought-to-an-algorithm-is-really-an-improvement-over-not-doing-anything)
26. [What are the hyperparameter tuning methods?](#what-are-the-hyperparameter-tuning-methods)
27. [How do we use probability in Machine Learning/Deep Learning framework?](#how-do-we-use-probability-in-machine-learningdeep-learning-framework)
28. [What are the differences and similarities between Ordinary Least Squares Estimation and Maximum Likelihood Estimation methods?](#what-are-the-differences-and-similarities-between-ordinary-least-squares-estimation-and-maximum-likelihood-estimation-methods)
29. [Do you suggest that treating a categorical variable as continuous variable would result in a better predictive model?](#do-you-suggest-that-treating-a-categorical-variable-as-continuous-variable-would-result-in-a-better-predictive-model)
30. [Considering the long list of machine learning algorithm, given a data set, how do you decide which one to use?](#considering-the-long-list-of-machine-learning-algorithm-given-a-data-set-how-do-you-decide-which-one-to-use)
31. [What is selection bias?](#what-is-selection-bias)
32. [What’s the difference between a generative and discriminative model?](#whats-the-difference-between-a-generative-and-discriminative-model)
33. [What cross-validation technique would you use on a time series dataset?](#what-cross-validation-technique-would-you-use-on-a-time-series-dataset)
34. [What is the difference between "long" and "wide" format data?](#what-is-the-difference-between-long-and-wide-format-data)
35. [Can you cite some examples where a false positive is important than a false negative, and where a false negative important than a false positive, and where both false positive and false negatives are equally important?](#can-you-cite-some-examples-where-a-false-positive-is-important-than-a-false-negative-and-where-a-false-negative-important-than-a-false-positive-and-where-both-false-positive-and-false-negatives-are-equally-important)
36. [Describe the difference between univariate, bivariate and multivariate analysis?](#describe-the-difference-between-univariate-bivariate-and-multivariate-analysis)
29. How do you deal with missing value in a data set?
30. How do you deal with imbalanced data?
31. How do you deal with high cardinality? 

## Linear Algebra

#### What are scalars, vectors, matrices, and tensors?

Scalars are single numbers and are an example of a 0th-order tensor. The notation $x \in \mathbb{R}$ states that the scalar value $x$ is an element of (or member of) the set of real-valued numbers, $\mathbb{R}$.

There are various sets of numbers of interest within machine learning. $\mathbb{N}$ represents the set of positive integers $(1,2,3, ...)$. $\mathbb{Z}$ represents the integers, which include positive, negative and zero values. $\mathbb{Q}$ represents the set of rational numbers that may be expressed as a fraction of two integers.

Vectors are ordered arrays of single numbers and are an example of 1st-order tensor.  An $n$-dimensional vector itself can be explicitly written using the following notation:

\begin{equation}
\boldsymbol{x}=\begin{bmatrix}
  \kern4pt x_1 \kern4pt \\
  \kern4pt x_2 \kern4pt \\
  \kern4pt \vdots \kern4pt \\
  \kern4pt x_n \kern4pt
\end{bmatrix}
\end{equation}

We can think of vectors as identifying points in space, with each element giving the coordinate along a different axis

One of the primary use cases for vectors is to represent physical quantities that have both a magnitude and a direction. Scalars are only capable of representing magnitudes.

Matrices are rectangular arrays consisting of numbers and are an example of 2nd-order tensors. If $m$ and $n$ are positive integers, that is $m, n \in \mathbb{N}$ then the $m \times n$ matrix contains $mn$ numbers, with $m$ rows and $n$ columns.

If all of the scalars in a matrix are real-valued then a matrix is denoted with uppercase boldface letters, such as $A \in \mathbb{R}^{m \times n}$. That is the matrix lives in a $m \times n$-dimensional real-valued vector space. 

Its components are now identified by two indices $i$ and $j$. $i$ represents the index to the matrix row, while $j$ represents the index to the matrix column. Each component of $A$ is identified by $a_{ij}$.

The full $m \times n$ matrix can be written as:

$$
\boldsymbol{A}=\begin{bmatrix}
   a_{11} & a_{12} & a_{13} & \ldots & a_{1n} \\
   a_{21} & a_{22} & a_{23} & \ldots & a_{2n} \\
   a_{31} & a_{32} & a_{33} & \ldots & a_{3n} \\
   \vdots & \vdots & \vdots & \ddots & \vdots \\
   a_{m1} & a_{m2} & a_{m3} & \ldots & a_{mn} \\
\end{bmatrix}
$$

It is often useful to abbreviate the full matrix component display into the following expression:

\begin{equation}
\boldsymbol{A} = [a_{ij}]_{m \times n}
\end{equation}

Where $a_{ij}$ is referred to as the $(i,j)$-element of the matrix $A$. The subscript of $m \times n$ can be dropped if the dimension of the matrix is clear from the context.

Note that a column vector is a size $m \times 1$ matrix, since it has $m$ rows and $1$ column. Unless otherwise specified all vectors will be considered to be column vectors.

Tensor is n-dimensional array. It encapsulates the scalar, vector and the matrix. For a 3rd-order tensor elements are given by $a_{ijk}$, whereas for a 4th-order tensor elements are given by $a_{ijkl}$.

#### How do you normalize a vector?

A vector is normalized when its norm is equal to one. To normalize a vector, we divide each of its elements by its norm or length (also known as magnitude). The norm of a vector is the square root of the sum of squares of the elements.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/norm_a_Vector.png?raw=true)

$$
\bar{x} = \frac{x}{\lVert x \rVert}
$$

For example,

$$
x = \begin{bmatrix} 2 \\ 1\\ 2\end{bmatrix}
$$

For example, the length (norm) of this vector is:

$$
\lVert x \rVert =\sqrt{2^{2} + 1^{2} + 2^{2}} = 3
$$

Therefore, if we normalize the vector $x$, we will get:

$$
\bar{x} = \begin{bmatrix} 2/3 \\ 1/3 \\ 2/3\end{bmatrix}
$$

#### What is Hadamard product of two matrices?
Hadamard product is also known as Element-wise Multiplication. It is named after French Mathematician, Jacques Hadamard. Elements corresponding to same row and columns of given vectors/matrices are multiplied together to form a new vector/matrix.

$$
    \begin{bmatrix} 
3 & 5 & 7 \\
4 & 9 & 8 
\end{bmatrix} \times \begin{bmatrix} 
1 & 6 & 3 \\
0 & 2 & 9 
\end{bmatrix} = \begin{bmatrix} 
3 \times 1 & 5 \times 6 & 7 \times 3 \\
4 \times 0 & 9 \times 2 & 8 \times 9 
\end{bmatrix} = \begin{bmatrix} 
3 & 30 & 21 \\
0 & 18 & 72 
\end{bmatrix}
$$

#### What is a scalar valued function?

A scalar valued function is a function that take one or more values but returns a single value. For example:

$$
f(x, y, z) = x^{2} + 2y z^{5}
$$

A $n$-variable scalar valued function acts as a map from the space $\mathbb{R}^{n}$ to the real number line $\mathbb{R}$. That is: $f:\mathbb{R}^{n} \to \mathbb{R}$

#### What is a vector valued function?

A vector valued function (also known as vector function) is a function where the domain is a subset of real numbers and the range is a vector. For example:

$$
r(t) = <2x+1, x^{2}+3>
$$

presents a function whose input is a scalar $t$ and whose output is a vector in $\mathbb{R}^{2}$

#### What is the gradient?

The gradient of a function $f$, denoted as $\nabla f$ is the collection of all its first-order partial derivatives into a vector. Here, $f$ is a scalar-valued (real-valued) multi-variable function $f:\mathbb{R}^{n}\to \mathbb{R}$.

$$
\nabla f(x_{1}, x_{2}, x_{3}, ...) = \begin{bmatrix} \dfrac{\partial f}{\partial x_{1}} \\[6pt] \dfrac{\partial f}{\partial x_{2}}\\[6pt] \dfrac{\partial f}{\partial x_{3}}\\[6pt] .\\.\\. \end{bmatrix}
$$

In particular, $\nabla f(x_{1}, x_{2}, x_{3}, ...)$ is a vector-valued function, which means it is a vector and we cannot take the gradient of a vector. 

It is very important to remember that the gradient of a function is only defined if the function is real-valued, that is, if it returns a scalar value. 

The most important thing to remember about the gradient is that the gradient of $f$, if evaluated at an input $(x_{0},y_{0})$, points in the direction of the steepest ascent. So, if you walk in that direction of the gradient, you will be going straight up the hill. Similarly, the magnitude of the vector $\nabla f(x_{0},y_{0})$ tells you what the slope of the hill is in that direction, meaning that if you walk in that direction, you will increase the value of $f$ at most rapidly.  

Note that the symbol $\nabla$ is referred to either as nabla or del. 

Note that the gradient of a vector-valued function is the same as obtaining the Jacobian of this function.

#### What is a Jacobian matrix?

Jacobian matrix is a matrix of all first-order partial derivatives of a vector-valued function. Suppose $f:\mathbb{R}^{n} \to \mathbb{R}^{m}$ is a function which takes as input the vector $x \in \mathbb{R}^{n}$ and produces as output the vector $f(x) \in \mathbb{R}^{m}$. Then, the Jacobian matrix J of $f$ is a $m \times n$ matrix:

$$
J = \begin{bmatrix}
\dfrac{\partial f}{\partial x_{1}} & \cdots &\dfrac{\partial f}{\partial x_{n}}
\end{bmatrix} =
\begin{bmatrix}
\dfrac{\partial f_{1}}{\partial x_{1}} &\cdots &\dfrac{\partial f_{1}}{\partial x_{n}} \\[6pt]
& \cdots & \\[6pt]
\dfrac{\partial f_{m}}{\partial x_{1}} &\cdots &\dfrac{\partial f_{m}}{\partial x_{n}} \\[6pt]
\end{bmatrix}
$$

Note that when $m=1$, the Jacobian is the same as gradient because it is a generalization of the gradient.

#### What is a Hessian matrix?

The hessian matrix is a square matrix of the second-order partial derivatives of a scalar-values (real-valued) multi-variable function $f:\mathbb{R}^{n}\to \mathbb{R}$.

If we have a scalar-valued multi-variable function $f(x_{1}, x_{2}, x_{3}, ...)$, its Hessian with respect to x, is the $n \times n$ matrix of partial derivatives:

$$
H_{f} \in \mathbb{R}^{n\times n}= \begin{bmatrix}
\dfrac{\partial^{2}f(x)}{\partial x_{1}^{2}} & \dfrac{\partial^{2}f(x)}{\partial x_{1} \partial x_{2}} & \cdots & \dfrac{\partial^{2}f(x)}{\partial x_{1} \partial x_{n}}\\[7pt]
\dfrac{\partial^{2}f(x)}{\partial x_{2} \partial x_{1}} & \dfrac{\partial^{2}f(x)}{\partial x_{2}^{2}} & \cdots & \dfrac{\partial^{2}f(x)}{\partial x_{2} \partial x_{n}} \\[7pt]
\vdots & \vdots & \cdots & \vdots \\[7pt]
\dfrac{\partial^{2}f(x)}{\partial x_{n} \partial x_{1}} & \dfrac{\partial^{2}f(x)}{\partial x_{n} x_{2}} & \cdots & \dfrac{\partial^{2}f(x)}{\partial x_{n}^{2}}
\end{bmatrix}
$$

Similar to the gradient, the Hessian is defined only when $f(x)$ is real-valued.

Hessian is NOT the gradient of the gradient!

Note that Hessian of a function $f:\mathbb{R}^{n}\to \mathbb{R}$ is the Jacobian of its gradient, i.e., $H(f(x)) = J(\nabla f(x))^{T}$.

#### What is an identity matrix?

identity matrix, $I \in \mathbb{R}^{n \times n}$, is a square matrix with ones on the diagonal and zeros everywhere else.

$$
    I_{ij} = \left\{ \begin{array}{ll}
         1 & \mbox{if $i=j$};\\
        0 & \mbox{if $i \neq j$}.\end{array} \right.
$$

It has the property that for all $A \in \mathbb{R}^{m \times n}$

$$
    AI = IA = A
$$

Generally the dimensions of $I$ are inferred from context so as to make matrix multiplication possible. 

#### What is the transpose of a matrix?

The transpose of a matrix results from "flipping" the rows and columns. Given a matrix $A \in \mathbb{R}^{m \times n}$, its transpose, written as $A^{T} \in \mathbb{R}^{n \times m}$, is the matrix whose entries are given by $\left( A_{ij}^{T} \right) = A_{ji}$.

The following properties of transposes are easily verified:
1. $\left(A^{T}\right)^{T} =A$
2. $\left(AB\right)^{T} = B^{T}A^{T}$
3. $\left(A + B\right)^{T} = A^{T} + B^{T}$

#### What is an inverse matrix?

The inverse of a square matrix $A$, sometimes called a reciprocal matrix, is a matrix $A^{-1}$ such that

$$
AA^{-1} = I = A^{-1}A
$$

where $I$ is the identity matrix. 
Note that for square matrices, the left inverse and right inverse are equal.

Non-square matrices do not have inverses by definition. Note that not all square matrices have inverses. A square matrix which has an inverse is called invertible or non-singular, and a square matrix without an inverse is called non-invertible or singular.

#### When does inverse of a matrix exist?

**Determine its rank**. In order for a square matrix $A$ to have an inverse $A^{-1}$, then $A$ must be full rank. The rank of a matrix is a unique number associated with a square matrix. The rank of a matrix is the number of non-zero eigenvalues
of the matrix. The rank of a matrix gives the dimensionality of the Euclidean space which can be used to represent this matrix. Matrices whose rank is equal to their dimensions are full rank and they are invertible. When the rank of a matrix is smaller than its dimensions, the matrix is not invertible and is called rank-deficient, singular, or multicolinear. For example, if the rank of an $n \times n$ matrix is less than $n$, the matrix does not have an inverse. 

**Compute its determinant**. The determinant is another unique number associated with a square matrix. The determinant det(A) of a square matrix A is the product of its eigenvalues. When the determinant for a square matrix is equal to zero, the inverse for that matrix does not exist.

$A, B \in \mathbb{R}^{n\times n}$ are non-singular. 
1. $(A^{-1})^{-1} = A$
2. $(AB)^{-1} = B^{-1}A^{-1}$
3. $(A^{-1})^{T} = (A^{T})^{-1}$

#### If inverse of a matrix exists, how to calculate it?

$$
\begin{split}
 Ax &= b\\
A^{-1}Ax &= A^{-1}b\\
I_{n}x &= A^{-1}b\\
x &= A^{-1}b  
\end{split}
$$

where $I_{n} \in \mathbb{R}^{n\times n}$

For example, for $2 \times 2$ matrix, the inverse is:

$$
    \begin{bmatrix}
    a & b \\
    c & d
    \end{bmatrix}^{-1} = \dfrac{1}{ad - bc}  \begin{bmatrix}
    d & -b \\
    -c & a
    \end{bmatrix}
$$

where $ad-bc$ is the determinant of this matrix. In other words, swap the positions of $a$ and $d$, put the negatives in front of $b$ and $c$ and divide everything by determinant. $AA^{-1} = I = A^{-1}A$ should be satisfied.

Now let's find the inverse of a bigger matrix, which is $3 \times 3$:

$$
    \begin{bmatrix}
    1 & 3 & 3 \\
    1 & 4 & 3 \\
    1 & 3 & 4
    \end{bmatrix}
$$

First, we write down the entries the matrix, but we write them in a double-wide matrix:

$$
    \begin{bmatrix}
    1 & 3 & 3 & | &  &  &  \\
    1 & 4 & 3 & | &  &  & \\
    1 & 3 & 4 & | &  &  & 
    \end{bmatrix}
$$

In the other half of the double-wide, we write the identity matrix:

$$
    \begin{bmatrix}
    1 & 3 & 3 & | & 1 & 0 & 0 \\
    1 & 4 & 3 & | & 0 & 1 & 0\\
    1 & 3 & 4 & | & 0 & 0 & 1
    \end{bmatrix}
$$

Now we'll do matrix row operations to convert the left-hand side of the double-wide into the identity. (As always with row operations, there is no one "right" way to do this. What follows are just one way. Your calculations could easily look quite different.)

$$
\begin{split}
    \begin{bmatrix}
    1 & 3 & 3 & | & 1 & 0 & 0 \\
    1 & 4 & 3 & | & 0 & 1 & 0\\
    1 & 3 & 4 & | & 0 & 0 & 1
    \end{bmatrix}&\underset{\overset{-r_{1}+r_{2}}{\longrightarrow}}{\overset{-r_{1}+r_{3}}{\longrightarrow}}
    \begin{bmatrix}
    1 & 3 & 3 & | & 1 & 0 & 0 \\
    0 & 1 & 0 & | & -1 & 1 & 0\\
    0 & 0 & 1 & | & -1 & 0 & 1
    \end{bmatrix}\\ &\overset{-3r_{2}+r_{1}}{\longrightarrow}
    \begin{bmatrix}
    1 & 0 & 3 & | & 4 & -3 & 0 \\
    0 & 1 & 0 & | & -1 & 1 & 0\\
    0 & 0 & 1 & | & -1 & 0 & 1
    \end{bmatrix}\\ &\overset{-3r_{3}+r_{1}}{\longrightarrow}
    \begin{bmatrix}
    1 & 0 & 0 & | & 7 & -3 & -3 \\
    0 & 1 & 0 & | & -1 & 1 & 0\\
    0 & 0 & 1 & | & -1 & 0 & 1
    \end{bmatrix}
    \end{split}
$$

Now that the left-hand side of the double-wide contains the identity, the right-hand side contains the inverse. That is, the inverse matrix is the following:

$$
\begin{bmatrix}
    7 & -3 & -3 \\
    -1 & 1 & 0\\
    -1 & 0 & 1
    \end{bmatrix}
$$

#### What is the determinant of a square matrix? How is it calculated? What is the connection of determinant to eigenvalues?

The determinant of a square matrix, denoted $det(A)$, is a function that maps matrices to real scalars, i.e., $det: \mathbb{R}^{n \times n} \to \mathbb{R}$. The determinant is equal to the product of all the eigenvalues of the matrix. The absolute value of the determinant can be thought of as a measure of how much multiplication by the matrix expands or contracts space. If the determinant is 0, then space is contracted completely along at least one dimension, causing it to lose all its volume. If the determinant is 1, then the transformation preserves volume.

The determinant is a real number, it is not a matrix. The determinant can be a negative number. The determinant only exists for square matrices. The determinant of a $1 \times 1$ matrix is that single value in the determinant. The inverse of a matrix will exist only if the determinant is not zero.

For example let's find a determinant of a $3 \times 3$ matrix:

$$
A = \begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33} 
\end{bmatrix}
$$

$$
\begin{split}
det(A) = |A| &=
a_{11} \begin{bmatrix}
a_{22} & a_{23} \\
a_{32} & a_{33}
\end{bmatrix}
- a_{12}\begin{bmatrix}
a_{21} &  a_{23} \\
a_{31} & a_{33} 
\end{bmatrix}
+ a_{13}\begin{bmatrix}
a_{21} & a_{22} \\
a_{31} & a_{32}
\end{bmatrix}\\
&= a_{11}\times \left(a_{22}a_{33}-a_{23}a_{32} \right)
-a_{12}\left(a_{21}a_{33} -a_{23}a_{31}  \right)
+a_{13} \left(a_{21}a_{32} - a_{22}a_{31}\right)
\end{split}
$$

It is the similar idea for $4\times 4$ and higher matrices.

You do not have to use the first row to compute the determinant. You can use any row or any column, as long as you know where to put the plus and minus signs.

$$
A = \begin{bmatrix}
+ & - & + & \cdots \\
- & + & - & \cdots \\
+ & - & + & \cdots \\
\vdots & \vdots & \vdots &  \cdots
\end{bmatrix}
$$

For example, 

$$
A = \begin{bmatrix}
1 & 0 & 2 & -1 \\
3 & 0 & 0 & 5 \\
2 & 1 & 4 & -3 \\
1 & 0  & 5 & 0
\end{bmatrix}
$$

The second column of this matrix has a lot of zeros.

$$
\begin{split}
det(A) = |A| &=
-0 \begin{bmatrix}
3  & 0 & 5 \\
2  & 4 & -3 \\
1  & 5 & 0
\end{bmatrix}
+0\begin{bmatrix}
1  & 2 & -1 \\
2  & 4 & -3 \\
1  & 5 & 0
\end{bmatrix}
-1\begin{bmatrix}
1 & 2 & -1 \\
3 & 0 & 5 \\
1 & 5 & 0
\end{bmatrix}
+0 \begin{bmatrix}
1 & 2 & -1 \\
3 & 0 & 5 \\
2 & 4 & -3 \\
\end{bmatrix}\\
&= -1\begin{bmatrix}
1 & 2 & -1 \\
3 & 0 & 5 \\
1 & 5 & 0
\end{bmatrix}\\
&=-1\left( 1\begin{bmatrix}
0 & 5 \\
5 & 0
\end{bmatrix}-2\begin{bmatrix}
3  & 5 \\
1  & 0
\end{bmatrix}+(-1)\begin{bmatrix}
3 & 0 \\
1 & 5 
\end{bmatrix} \right)\\
&=-1(-25+10-15)\\
&=30
\end{split}
$$

#### What is a norm? What is $L^{1}$, $L^{2}$ and $L^{\infty}$ norm? What are the conditions a norm has to satisfy?

Sometimes, we need to measure the size of a vector (length of the vector). In machine learning, we usually measure the size of vectors using a function called a norm. Formally, assuming $x$ is a vector and $x_{i}$ is its $i$th-element, the $L^{p}$ norm is given by

$$
    \lvert x \rvert_{p} = \left(\sum_{i=1}\lvert x_{i} \rvert^{p}  \right)^{1/p}
$$

for $p \in \mathbb{R}$, and $p \geq 1$.

* $L^{1}$ is known as Manhattan Distance (norm).
* $L^{2}$ is known as Euclidean Distance (norm) which gives the magnitude of a vector. However, confusion is that the Frobenius norm (a matrix norm) is also sometimes called the Euclidean norm.
* $L^{\infty} = \underset{i}{\max} \lvert x_{i} \rvert$ also known as the max norm (sup norm). This norm simplifies to the absolute value of the element with the largest magnitude in the vector.

The higher the norm index, the more it focuses on large values and neglects small ones. This is why the Root Mean Squared Error (RMSE, which corresponds to Euclidean norm) is more sensitive to outliers than Mean Absolute Error (MAE which corresponds to Manhattan norm). But when outliers are exponentially rare (like in a bell-shaped curve), the RMSE performs very well and is generally preferred.

Norms, including the $L^{p}$ norm, are functions mapping vectors to non-negative values. On an intuitive level, the norm of a vector $x$ measures the distance from the origin to the point $x$. More rigorously, a norm is any function $f$ that satisﬁes the following properties

* $f(x) = 0 \Rightarrow x = 0$ (Definiteness)
* $f(x) \geq 0$ (non-negativity)
* $f(x + y) \leq f (x) + f(y)$ (the triangle inequality)
* $\forall \alpha \in \mathbb{R}, f( \alpha x) = \lvert \alpha \rvert f(x)$ (homogenity)

#### What is Frobenius norm?

Sometimes we may also wish to measure the size of a matrix. In the context of deep learning, the most common way to do this is with the otherwise obscure Frobenius norm.

The Frobenius norm, sometimes also called the Euclidean norm (a term unfortunately also used for the vector $L^{2}$-norm), is matrix norm of an $m \times n$ matrix $A$ defined as the square root of the sum of the squares of its elements:

$$
\lvert A \rvert_{F} = \sqrt{\sum_{i=1}^{m}\sum_{j=1}^{n} a_{ij}^{2}}
$$

which is analogous to the $L^{2}$-norm of a vector

#### What is a diagonal matrix?

Diagonal matrices consist mostly of zeros and have nonzero entries only along the main diagonal. Formally, a matrix $D$ is diagonal if and only if $D_{i,j}= 0$ for all $i \neq j$.

$$
    D_{ij} = \left\{ \begin{array}{ll}
         d_{i} & \mbox{if $i=j$};\\
        0 & \mbox{if $i \neq j$}.\end{array} \right.
$$

Identity matrix, where all the diagonal entries are 1 is an example of a diagonal matrix.  Clearly, $I = \text{diag}(1,1,1,...,1)$.

Not all diagonal matrices need be square. It is possible to construct a rectangular diagonal matrix.

$$
\begin{bmatrix} 
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix} \begin{bmatrix} 
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0
\end{bmatrix} \begin{bmatrix} 
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1 \\
0 & 0 & 0
\end{bmatrix}
$$

Entries in $i=j$ can technically be called the "main diagonal" of the rectangular matrix, though the diagonal of such a matrix is not necessarily as "useful" as it is in a square matrix. 

#### What is a symmetric matrix?

A symmetric matrix is any matrix that is equal to its own transpose:

$$
A = A^{T}
$$

It is anti-symmmetric if $A = -A^{T}$

#### What is the trace of a matrix?

The trace operator gives the sum of all the diagonal entries of a matrix:

$$
Tr(A) = \sum_{i} A_{i,i}
$$

It is easy to show that the trace is a linear map, so that

$$
Tr(\lambda A) = \lambda Tr(A) = \lambda \sum_{i} A_{i,i}
$$

and 

$$
Tr(A + B) = Tr(A) + Tr(B)
$$

The trace operator is invariant to the transpose operator: $Tr(A) = Tr(A^{T})$. 

#### How to write Frobenius norm of a matrix A in terms of trace?

The trace operator provides an alternative way of writing the Frobenius norm of a matrix:

$$
\lvert A \rvert_{F} = \sqrt{tr(A A^{T})} = \sqrt{tr(A^{T} A)}
$$

#### What is eigendecomposition, eigenvectors and eigenvalues? How to find eigenvalues of a matrix?

In linear algebra, eigendecomposition or sometimes spectral decomposition is the factorization of a matrix into a canonical form, whereby the matrix is represented in terms of its eigenvalues and eigenvectors. Only diagonalizable matrices can be factorized in this way. The eigen-decomposition can also be use to build back a matrix from it eigenvectors and eigenvalues. For details look [here](#what-is-spectral-decomposition).

Given a $p \times p$ matrix $A$, the real number $u$ and the vector $v$ are an eigenvalue and corresponding eigenvector of $A$ if

$$
Av = uv
$$

Here, $u$ is a scalar (which may be either real or complex).  Any value of $u$ for which this equation has a solution is known as an eigenvalue of the matrix $A$. It is sometimes also called the characteristic value.  The vector, v, which corresponds to this value is called an eigenvector.  The eigenvalue problem can be rewritten as 

The eigenvalue problem can be rewritten as:

$$
\begin{split}
Av &= uv \\
Av - uIv &= 0 \\
(A-uI)v &= 0 \\
\end{split}
$$

where $I$ is the $p \times p$ identity matrix. Now, in order for a non-zero vector $v$ to satisfy this equation, $A − uI$ must not be invertible, i.e.,

$$
\text{det} (A - uI) = 0
$$

This equation is known as the characteristic equation of A, and the left-hand side is known as the characteristic polynomial, and is an $p$-th order polynomial in $u$ with $p$ roots. These roots are called the eigenvalues of $A$.  We will only deal with the case of $p$ distinct roots, though they may be repeated.  For each eigenvalue there will be an eigenvector for which the eigenvalue equation is true.  

#### What is the trace of a scalar?

A scalar is its own trace $a=Tr(a)$

#### What is Spectral Decomposition?

Spectral decomposition, sometimes called _eigendecomposition_, recasts a real symmetric $p \times p$ matrix $A$ with its eigenvalues $u_{1}, u_{2}, \ldots, u_{p}$ and corresponding orthonormal eigenvectors $v_{1}, v_{2}, \ldots, v_{p}$, then, 

$$
\begin{split}
A &= \underbrace{\begin{bmatrix} \uparrow & \uparrow & \ldots & \uparrow \\
v_{1} & v_{2} & \ldots &  v_{p} \\
\downarrow & \downarrow & \ldots & \downarrow \\
\end{bmatrix}}_{\text{Call this Q}}\underbrace{\begin{bmatrix}
u_{1} & 0 & \ldots & 0 \\
0 & u_{2} & \ldots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \ldots & u_{p}\\
\end{bmatrix}}_{\Lambda}\underbrace{\begin{bmatrix}
\leftarrow & v_{1} & \rightarrow \\
\leftarrow & v_{2} & \rightarrow \\
\ldots & \ldots & \ldots \\
\leftarrow & v_{p} & \rightarrow \\
\end{bmatrix}}_{Q^{T}} \\
&= Q \Lambda Q^{T}
\end{split}
$$

or

$$
A = \sum_{i=1}^{p} u_{i} \mathbf{v}_{i} \mathbf{v}_{i}^{T}
$$

#### What do positive definite, positive semi-definite and negative definite/negative semi-definite mean?

A matrix $A$ is positive semi-definite if it is symmetric and all its eigenvalues are non-negative. If all eigenvalues are strictly positive then it is called a positive definite matrix.

A square symmetric matrix $A \in  \mathbb{R}^{n \times n}$ is positive semi-definite if 

$$
v^{T} A v \geq 0,\,\,\, \forall v \in \mathbb{R}^{n \times 1}
$$ 

and positive definite if the inequality holds with equality only for vectors $v=0$, i.e., $v^{T} A v > 0,\,\,\, \forall v \in \mathbb{R}^{n \times 1}$.

A square symmetric matrix $A \in  \mathbb{R}^{n \times n}$ is negative semi-definite if

$$
v^{T} A v \leq 0,\,\,\, \forall v \in \mathbb{R}^{n \times 1}
$$ 

and negative definite if the inequality holds with equality only for vectors $v=0$, i.e., $v^{T} A v < 0,\,\,\, \forall v \in \mathbb{R}^{n \times 1}$

Positive (semi)definite and negative (semi)definite matrices together are called definite matrices. A symmetric matrix that is not definite is said to be indefinite. 

A symmetric matrix is positive semi-definite if and only if all eigenvalues are non-negative. It is negative semi-definite if and only if all eigenvalues are non-positive. It is positive definite if and only if all eigenvalues are positive. It is negative definite if and only if all eigenvalues are negative.

The matrix $A$ is positive sem-definite if any only if $−A$ is negative semi-definite, and similarly a matrix $A$ is positive definite if and only if $−A$ is negative definite.

Now, let's see how we can use the quadratic form to check the positive definiteness:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/pd_nd_psd_nsd.png?raw=true)

To check if the matrix is positive definite/positive semi-definite/negative definite/negative semi-definite or not, you just have to compute the above quadratic form and check if the value is strictly positive/positive/strictly negative or negative.


Let's prove some of the statements above for positive definite matrix. 

* If a real symmetric matrix is positive definite, then every eigenvalue of the matrix is positive.
  Let's say here $v \in \mathbb{R}^{n}$ is the eigenvector and $u \in \mathbb{R}$ is eigenvalue. Using eigenvalue equation,
  
  $$
\begin{split}
A v = u v &\Rightarrow v^{T} A v = u \left(v^{T} v \right)\\
&\Rightarrow u = \dfrac{v^{T} A v}{v^{T} v} = \dfrac{v^{T} A v}{\left\Vert v \right\Vert_{2}^{2}}
\end{split}
$$
  
  Since $A$ is positive definite, its quadratic form is positive, i.e., $v^{T} A v > 0$. $v$ is a nonzero vector as it is an eigenvector. Since $\left\Vert v \right\Vert_{2}^{2}$ is positive, we must have $u$ is positive, which is $u > 0$.

* If every eigenvalue of a real symmetric matrix is positive, then the matrix is positive definite.

  By the spectral theorem, a real symmetric matrix has an eigenvalue decomposiiton, so,
  
  $$
  A = Q \Lambda Q^{T}
  $$
  
  For the quadratic function defined by $A$:
  
  $$
  v^{T} A v = \underbrace{v^{T} Q}_{y^{T}} \Lambda \underbrace{Q^{T} v}_{y} = \sum_{i=1}^{n} u_{i}y_{i}^{2}
  $$
  
  Since eigenvalues $u_{i}$'s are positive and $y_{i}^{2} > 0$, this summation is always positive, therefore, $v^{T} A v > 0$.

### How to make a positive definite matrix with a matrix that’s not symmetric?

The problem with definite matrices is that they are not always symmetric. However, we can simply multiply the matrix that’s not symmetric by its transpose and the product will become symmetric, square, and positive definite!

Let's say the matrix $B \in \mathbb{R}^{m\times n}$. Then, $B^{T}B \in  \mathbb{R}^{n\times n}$ which is a square matrix in real space. If $v^{T}B^{T}Bv = \left( Bv\right)^{T}\left( Bv\right) = \left\Vert Bv \right\Vert_{2}^{2} > 0$, then $B^{T}B$ is positive definite matrix.


## Numerical Optimization

#### Describe convex function

A function $f(x): M \rightarrow \mathbb{R}$, defined on a nonempty subset $M$ of $\mathbb{R}^{n}$ and taking real values, is convex on an interval $[a,b]$ if for any two points $x_1$ and $x_2$ in $[a,b]$ and any lambda where $0< \lambda < 1$,

* the domain M of the function is convex, meaning it is a convex set if it contains all convex combinations of any two points within it.

and 

* $$
f[\lambda x_{1} + (1 - \lambda) x_{2}] \leq \lambda f(x_{1}) + (1 - \lambda) f(x_{2}) 
$$

If $f(x)$ has a second derivative in $[a,b]$, then a necessary and sufficient condition for it to be convex on on the interval $[a,b]$ is that the second derivative $f^{''}(x) \geq 0$ for all $x$ in $[a,b]$. However, the converse need not be true.

The prototypical convex function is shaped something like the letter U.

If the inequality above is strict for all $x_{1}$ and $x_{2}$, then $f(x)$ is called strictly convex.

An inequality is strict if replacing any "less than" and "greater than" signs with equal signs never gives a true expression. For example, $a \leq b$ is not strict, whereas $a < b$ is.

Some convex function examples are shown below:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/convex_func_examples.png?raw=true)

By contrast, the following function is not convex. Notice how the region above the graph is not a convex set:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/nonconvex_func_example.png?raw=true)

A strictly convex function has exactly one local minimum point, which is also the global minimum point. The classic U-shaped functions are strictly convex functions. However, some convex functions (for example, straight lines) are not U-shaped.

If the functions $f$ and $g$ are convex, then any linear combination $a f + b g$ where $a$, $b$ are positive real numbers is also convex.

$f$ is convex if $−f$ is concave.

The introduced concept of convexity has a simple geometric interpretation. Geometrically, the line segment connecting $(x_{1}, f(x_{1}))$ to $(x_{2}, f(x_{2}))$ must sit above the graph of $f$ and never cross the graph itself.

If a function is convex, the midpoint $B$ of each chord $A_{1}A_{2}$ lies above the corresponding point $A_{0}$ of the graph of the function or coincides with this point.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/graphic_convex.png?raw=true)


## Set Theory
![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/set_theory.gif?raw=true)

#### What is a random experiment?
A random experiment is an experiment or a process for which the outcome cannot be predicted with certainty.

#### What is a sample space?
The sample space (mostly denoted by $S$) of a random experiment is the set of all possible outcomes. $S$ is called the certain event.

Let $A$ be a set. The notation $x \in A$ means that $x$ belongs to $A$. 

#### What is an empty set?
In mathematics, the empty set, mostly denoted by $\emptyset$, is a set with no elements; its size or cardinality (count of elements in a set) is zero. Empty set is called the impossible event. 

Let $A$ be a set. 
* The intersection of any set with the empty set is the empty set, i.e., $A \cap \emptyset = \emptyset$.
* The union of any set with the empty set is the set we started with, i.e., $A \cup \emptyset = A$.
* The complement of the empty set is the universal set (U) for the setting that we are working in, i.e., $\emptyset^C = U - \emptyset = U$. Also, the complement of $U$ is the empty set: $U^{c} =  U - U = \emptyset$.
* The empty set is a subset of any set.

#### What is an event?
An event (mostly denoted by $E$) is a subset of the sample space. We say that $E$ has occured if the observed outcome $x$ is an element of $E$, that is $x \in E$

**Examples**:
* Random experiment: toss a coin, sample sample $S = \{ \text{heads}, \text{tails}\}$
* Random experiment: roll a dice, sample sample $S = \{1,2,3,4,5,6\}$

#### What are the operations on a set?
When working with events, __intersection__ means "and", and __union__ means "or".

$$
P(A \cap B) = P(\text{A and B}) = P(A, B)
$$

$$
P(A \cup B) = P(\text{A or B})
$$

#### What is mutually exclusive (disjoint) events?
The events in the sequence $A_{1}, A_{2}, A_{3}, \ldots$ are said to be mutually exclusive events if $E_{i} \cap E_{j} = \emptyset\text{ for all }i \neq j$ where $\emptyset$ represents the empty set. 

In other words, the events are said to be mutually exclusive if they do not have any outcomes (elements) in common, i.e., they are pairwise disjoint. 

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/mutually_exclusive.png?raw=true)

For example, if events $A$ and $B$ are mutually exclusive:

$$
P(A \cup B) = P(A) + P(B)
$$

#### What is a non-disjoint event?
Disjoint events, by definition, can not happen at the same time. A synonym for this term is mutually exclusive. Non-disjoint events, on the other hand, can happen at the same time. For example, a student can get grade A in Statistics course and A in History course at the same time.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/nondisjoint_events.png?raw=true)

For example, if events $A$ and $B$ are non-disjoint events, the probability of A or B happening (union of these events) is given by:

$$
P(A \cup B) = P(A) + P(B) - P(A \cap B)
$$

This rule is also called Addition rule.

#### What is an independent event?

An independent event is an event that has no connection to another event's chances of happening (or not happening). In other words, the event has no effect on the probability of another event occurring.

Let's say that we have two events, $A$ and $B$ and they are independent. Intersection of these two events has:

$$
P(A\cap B) = P(A)P(B)
$$

This rule is also called multiplication rule.

Therefore, their union is:

$$
P(A\cup B) = P(A) + P(B) - P(A\cap B) =  P(A) + P(B) - P(A)P(B)
$$

#### What is exhaustive events?
When two or more events form the sample space ($S$) collectively, then it is known as collectively-exhaustive events. The $n$ events $A_{1}, A_{2}, A_{3}, \ldots, A_{n}$ are said to be exhaustive if $A_{1} \cup A_{2} \cup A_{3} \cup \ldots \cup A_{n} = S$. 

#### What is Inclusion-Exlusive Principle?
* For $n=2$ events:

$$
P(A \cup B) = P(A) + P(B) - P(A \cap B)
$$

Let's prove this principle:

$$
\begin{split}
P(A \cup B) &= P(A \cup (B-A))\\
&= P(A) + P(B-A)\\
&= P(A) + P(B) - P(A \cap B)
\end{split}
$$

* For events $A_{1}, A_{2}, A_{3}, \ldots, A_{n}$ in a probability space:

$$
\begin{split}
P\left(\cup_{i=1}^{n} A_{i}\right) =\sum_{i=1}^{n} P(A_{i}) &- \sum_{1 \leq i \leq j \leq n} P(A_{i} \cap A_{j})\\
&+ \sum_{1 \leq i \leq j \leq k \leq n} P(A_{i} \cap A_{j} \cap A_{k}) - \ldots \\
& + \left(-1\right)^{n-1} P\left(\cap_{i=1}^{n} A_{i}\right)
\end{split}
$$

## Probability 

#### What is a probability?
We assign a probability measure $P(A)$ to an event $A$. This is a value between $0$ and $1$ that shows how likely the event is. If $P(A)$ is close to $0$, it is very unlikely that the event $A$ occurs. On the other hand, if $P(A)$ is close to $1$, $A$ is very likely to occur. 

#### What are the probability axioms?

* **Axiom 1** For any event $A$, $P(A) \geq 0$
* **Axiom 2** Probability of the sample space is $P(S)=1$ 
* **Axiom 3** If $A_{1}, A_{2}, A_{3}, \ldots$ are disjoint (mutually exclusive) (even countably infinite) events, meaning that they have an empty intersection, the probability of the union of the events is the same as the sum of the probabilities: $P(A_{1} \cup A_{2} \cup A_{3} \cup \ldots) = P(A_{1}) + P(A_{2}) + P(A_{3}) + \dots$.

#### What is a random variable?
A random variable is a variable whose values depend on all the possible outcomes of a natural phenomenon. In short, a random variable is a quantity produced by a random process. Each numerical outcome of a random variable can be assigned a probability. There are two types of random variables, discrete and continuous. 

A discrete random variable is one which may take on a finite or countably infinite number of possible values such as 0,1,2,3,4,... Discrete random variables are usually (but not necessarily) counts. If a random variable can take only a finite number of distinct values, then it must be discrete. Examples of discrete random variables include the number of children in a family, the Friday night attendance at a cinema, the number of patients in a doctor's surgery, the number of defective light bulbs in a box of ten.

A continuous random variable is one which takes on an uncountably infinite number of possible values. Continuous random variables are usually measurements. Examples include height, weight, the amount of sugar in an orange, the time required to run a mile.

#### What is a probability distribution?

In probability theory and statistics, a probability distribution is a mathematical function that provides the probabilities of occurrence of different possible outcomes in an experiment. In more technical terms, the probability distribution is a description of a random phenomenon in terms of the probabilities of events. For instance, if the random variable $X$ is used to denote the outcome of a coin toss ("the experiment"), then the probability distribution of $X$ would take the value $0.5$ for $X = \text{heads}$, and $0.5$ for $X = \text{tails}$ (assuming the coin is fair). Examples of random phenomena can include the results of an experiment or survey.

Discrete probability functions are referred to as probability mass functions and continuous probability functions are referred to as probability density functions. 

#### What is a probability mass function? What are the conditions for a function to be a probability mass function?

Discrete probability function is referred to as probability mass function (pms). It is a function that gives the probability that a discrete random variable is exactly equal to some value. The mathematical definition of a discrete probability function, $p(x)$, is a function that satisfies the following properties:

* The probability that $x$ can take a specific value is $p(x)$. That is

$$
P[X=x]=p(x)=p_{x}
$$

* $p(x)$ is non-negative for all real $x$.

* The sum of $p(x)$ over all possible values of $x$ is $1$, that is

  $$
  \sum_{j}p_{j} = 1
  $$ 
  
  where $j$ represents all possible values that $x$ can have and $p_{j}$ is the probability at $x_j$.

  One consequence of properties 2 and 3 is that $0 \leq p(x) \leq 1$.

A discrete probability function is a function that can take a discrete number of values (not necessarily finite). This is most often the non-negative integers or some subset of the non-negative integers. There is no mathematical restriction that discrete probability functions only be defined at integers, but in practice this is usually what makes sense. For example, if you toss a coin 6 times, you can get 2 heads or 3 heads but not 2 1/2 heads. Each of the discrete values has a certain probability of occurrence that is between zero and one. That is, a discrete function that allows negative values or values greater than one is not a probability function. The condition that the probabilities sum to one means that at least one of the values has to occur.

#### What is a probability density function? What are the conditions for a function to be a probability density function?

Continuous probability function is referred to as probability density function (pdf). It is a function of a continuous random variable, whose integral across an interval gives the probability that the value of the variable lies within the same interval. Unlike discrete random variable, the probability for a given continuous variable can not be specified directly; instead, it is calculated as an integral (area under curve) for a tiny interval around the specific outcome.

The mathematical definition of a continuous probability function, $f(x)$, is a function that satisfies the following properties:

* The probability that x is between two points a and b is

$$
p[a \leq x \leq b]=\int_{a}^{b} f(x)dx
$$

* It is non-negative for all real $x$.

* The integral of the probability function is one, that is

$$
\int_{\infty}^{\infty} f(x)dx = 1
$$

#### What is a joint probability distribution? What is a marginal probability? Given the joint probability function, how will you calculate it?

In general, if $X$ and $Y$ are two random variables, the probability distribution that defines their simultaneous behavior is called a joint probability distribution, shown as $P(X =x, Y = y)$. If $X$ and $Y$ are discrete, this distribution can be
described with a _joint probability mass function_. If $X$ and $Y$ are continuous, this distribution can be described with a _joint probability density function_. If we are given a joint probability distribution for $X$ and $Y$ , we can obtain the individual probability distribution for $X$ or for $Y$ (and these are called the _Marginal Probability Distributions_).

Note that when there are two random variables of interest, we also use the term _bivariate probability distribution_ or _bivariate distribution_ to refer to the joint distribution.

The joint probability mass Punction of the discrete random variables $X$ and $Y$, denoted as $P_{XY} (x, y)$, satisfies:

* $P_{XY} (x, y) \geq 0$ for all x, y
* $\sum_{x} \sum_{y} P_{XY} (x, y) = 1$
* $P_{XY} (x, y) = P(X = x, Y = y)$

If $X$ and $Y$ are discrete random variables with joint probability mass function $P_{XY} (x, y)$, then the marginal probability mass functions oP $X$ and $Y$ are,

$$
P_{X} (x) = P(X=x) = \sum_{y_{j} \in \mathbb{R}_{y}} P_{XY}(X=x,Y=y_{j}) = \sum_{y} P_{XY} (x, y)
$$

and

$$
P_{Y} (y) = P(Y=y) = \sum_{x_{i} \in \mathbb{R}_{x}} P_{XY}(X=x_{i},Y=y) = \sum_{x} P_{XY} (x, y)
$$

where the sum for $P_{X} (x)$ is over all points in the range of $(X, Y)$ for which $X = x$ and the sum for $P_{Y} (y)$ is over all points in the range of $(X, Y)$ for which $Y = y$.

A joint probability density function for the continuous random variable $X$ and $Y$, denoted as $f_{XY} (x, y)$, satisfies the following properties:

* $f_{XY} (x, y) \geq 0$ for all x, y
* $\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f_{XY} (x, y) dx dy = 1$
* For any region \$\mathbb{R}$ of 2-D space:

$$
P((X, Y) \in \mathbb{R}) = \int_{\mathbb{R}} f_{XY} (x, y) dx dy
$$

If $X$ and $Y$ are continuous random variables with joint probability density function $f_{XY} (x, y)$, then the marginal density functions for $X$ and $Y$ are:

$$
f_{X} (x) = \int_{y} f_{XY} (x, y) dy
$$

and

$$
f_{Y} (y) = \int_{x} f_{XY} (x, y) dx
$$

where the first integral is over all points in the range of $(X, Y)$ for which $X = x$, and the second integral is over all points in the range of $(X, Y)$ for which $Y = y$.

The joint cumulative distribution function (CDF) of two random variables $X$ and $Y$ is defined as:

$$
\begin{split}
F_{XY}(x,y)&=P(X \leq x, Y \leq y) \\
&= P\big((X \leq x)\text{ and }(Y\leq y)\big)=P\big((X \leq x)\cap(Y\leq y)\big).
\end{split}
$$

The joint CDF satisfies the following properties:

* $F_X(x)=F_{XY}(x, \infty)$, for any $x$ (marginal CDF of $X$)
* $F_Y(y)=F_{XY}(\infty,y)$, for any $y$ (marginal CDF of $Y$)
* $F_{XY}(\infty, \infty)=1$
* $F_{XY}(-\infty, y)=F_{XY}(x,-\infty)=0$
* $P(x_1<X \leq x_2, \hspace{5pt} y_1<Y \leq y_2)= F_{XY}(x_2,y_2)-F_{XY}(x_1,y_2)-F_{XY}(x_2,y_1)+F_{XY}(x_1,y_1)$
* If X and Y are independent, then $F_{XY}(x,y)=F_X(x)F_Y(y)$. 

#### What is conditional probability? Given the joint probability function, how will you calculate it?

Let's say we have two events, $A$ and $B$. The conditional probability of an event $B$ is the probability that the event will occur given the knowledge that an event $A$ has already occurred. This probability is written $P(B \mid A)$, notation for the probability of $B$ given $A$.  In the case where events $A$ and $B$ are independent (where event $A$ has no effect on the probability of event $B$), the conditional probability of event $B$ given event $A$ is simply the probability of event $B$, that is $P(B)$.

$$
P(B \mid A) = P(B)
$$

Because $\frac{P(A \cap B) = P(A) \times P(B)$ when $A$ and $B$ are independent events.

However, If events $A$ and $B$ are not independent, then the probability of the intersection of $A$ and $B$ (the probability that both events occur) is defined by $P(A\text{ and }B) = P(A \cap B) = P(A)P(B \mid A)$, which $P(A\text{ and }B)$ is the joint probability. Intuitively it states that the probability of observing events $A$ and $B$ is the probability of observing $A$, multiplied by the probability of observing $B$, given that you have observed $A$.

From this definition, the conditional probability $P(B \mid A)$ is easily obtained by dividing by $P(A)$:

$$
P(B \mid A) = \dfrac{P(A \cap B)}{P(A)} 
$$

Note that this expression is only valid when $P(A)$ is greater than 0.

Technically speaking, when you condition on an event happening, you are entering the universe where that event has taken place. Mathematically, if you condition on $A$, then $A$ becomes your new sample space. In the universe where $A$ has taken place, all axioms of probability still hold! In particular,

* __Axiom 1:__ For any event $B$, $P(B \mid A) \geq 0$.
* __Axiom 2:__ Conditional probability of $A$ given $A$ is 1, i.e., $P(A \mid A)=1$.
* __Axiom 3:__ If $B_{1}, B_{2}, B_{3}, \ldots $ are disjoint events, then $P(B_{1} \cup B_{2} \cup B_{3} \cup \ldots \mid A) = P(B_{1} \mid A) + P(B_{2} \mid A) + P(B_{3} \mid A) + \dots $

#### State the Chain rule of conditional probabilities.

To calculate the probability of the intersection of more than two events, the conditional probabilities of all of the preceding events must be considered. In the case of three events, $A$, $B$, and $C$, the probability of the intersection $P(A\text{ and }B\text{ and }C) = P(A)P(B \mid A)P(C \mid A\text{ and }B)$, which we call the _Chain Rule_. Here is the general form of the Chain Rule when $n$ events are given:

$$
P(A_{1} \cap A_{2} \cap \ldots \cap A_{n}) = P(A_{1})P(A_{2} \mid A_{1})P(A_{3} \mid A_{2}, A_{1}) \ldots P(A_{n} \mid A_{n-1} A_{n-2} \cdots A_{1})
$$

#### What are expectation, variance and covariance?
In probability, the average value of some random variable X is called the **expected value** or the expectation, denoted by $E(x)$.

Suppose $X$ is a discrete random variable that takes values $x_{1}, x_{2}, . . . , x_{n}$ with probabilities $p(x_{1}), p(x_{2}), . . . , p(x_{n})$. The expected value of $X$ is defined by:

$$
E(X) = \sum_{j=1}^{n}  x_{j}p(x_{j}) = x_{1}p(x_{1}) + x_{2}p(x_{2})  + . . . + x_{n}p(x_{n}).
$$

Let $X$ be a continuous random variable with range $[a, b]$ and probability density function $f(x)$. The expected value of $X$ is defined by

$$
E(X) = \int_{a}^{b} xf(x) dx.
$$

In probability, the **variance** of some random variable $X$, denoted by $Var(X)$ is a measure of how much values in the distribution vary on average with respect to the mean. Variance is calculated as the average squared difference of each value in the distribution from the expected value. Or the expected squared difference from the expected value.

$$
\begin{split}
Var(X) &= E\left[\left(X - E[X]\right)^{2}\right] \\
&= E\left[X^{2}-2XE(X) + \left(E(X) \right)^{2}\right] \\
&= E(X^{2}) - 2E(X)E(X) + \left(E(X) \right)^{2}\\
&= E(X^{2}) - \left(E(X) \right)^{2}
\end{split}
$$

In probability, **covariance** is the measure of the joint probability for two random variables. It describes how the two variables change (vary) together. It’s similar to variance, but where variance tells you how a single variable varies, covariance tells you how two variables vary together. It is denoted as the function $cov(X, Y)$, where $X$ and $Y$ are the two random variables being considered.

$$
cov(X, Y) = E\left[\left(X - E[X]\right) \left(Y - E[Y]\right)\right] =  E(XY)- E(X)E(Y)
$$

Note that $Var(X) = cov(X, X) = E\left[(X - E[X])^{2}\right]$. 

The sign of the covariance can be interpreted as whether the two variables increase together (positive) or decrease together (negative). The magnitude of the covariance is not easily interpreted. A covariance value of zero indicates that both variables are completely independent.

The covariance can be normalized to a score between $-1$ and $1$ to make the magnitude interpretable by dividing it by the standard deviation of X and Y. The result is called the correlation of the variables, also called the _Pearson correlation coefficient_, named for the developer of the method, Karl Pearson.

$$
corr(X, Y) = \rho_{X, Y}= \frac{cov(X,Y)}{\sigma_{X}\sigma_{Y}}
$$

where $\sigma_{X} = \sqrt{Var(X)}$ and $\sigma_{Y} = \sqrt{Var(Y)}$.

As one can tell easily that correlation is just the covariance normalized.
 
The covariance is especially useful when looking at the variance of the sum of two random 'correlated' variates, since

$$
Var(X+Y) = Var(X)+ Var(Y) + 2cov(X,Y)
$$

If the variables are uncorrelated (that is, $cov(X, Y)=0$), then

$$
Var(X+Y) = Var(X) + Var(Y).
$$

In general,

$$
Var\left( \sum_{i=1}^{n} X_i \right)= \sum_{i,j=1}^{N}  cov(X_{i},X_{j}) = \sum_{i=1}^{n} Var( X_i) + \sum_{i \neq j} cov(X_{i},X_{j}).
$$

If for each $i \neq j$, $X_i$ and $X_j$ are uncorrelated, in particular if the $X_i$ are pairwise independent (that is, $X_i$ and $X_j$ are independent whenever $i \neq j$), then,

$$
Var\left( \sum_{i=1}^{n} X_i \right)=  \sum_{i=1}^{n} Var( X_i) .
$$

The covariance is symmetric by definition since

$$
cov(X,Y)=cov(Y,X). 
$$

#### What is the covariance for a vector of random variables?

A random vector is a vector of random variables:

$$
\mathbf{X} = \begin{bmatrix} X_{1} \\ X_{2}\\ \vdots \\ X_{n} \end{bmatrix}
$$

If $X$ is a random vector, the covariance matrix of $X$, denoted by $\Sigma$, is then given by:

$$
\begin{split}
\Sigma = cov(\mathbf{X}) &= E\left[ \left( \mathbf{X} - E(\mathbf{X}) \right) \left( \mathbf{X} - E(\mathbf{X}) \right)^{T} \right]\\
&= E\left[\mathbf{X}\mathbf{X}^{T} \right] - E[\mathbf{X}]\left(E[\mathbf{X}]\right)^{T}
\end{split}
$$

and defined as

$$
\Sigma = cov(\mathbf{X}) = 
\begin{bmatrix} Var(X_{1}) & cov(X_{1},X_{2}) & \ldots & cov(X_{1},X_{n}) \\
cov(X_{2}, X_{1}) & Var(X_{2}) & \ldots & cov(X_{2},X_{n}) \\
\vdots & \vdots & \ddots & \vdots \\
cov(X_{n}, X_{1}) & cov(X_{n}, X_{2}) & \ldots & Var(X_{n}) \\
\end{bmatrix}
$$

As we previously mentioned, covariance matrix is symmetric, meaning that $\Sigma_{i,j} = \Sigma_{j,i}$.

$$
\Sigma_{i,j} = cov(X_{i}, X_{j}) = E\left[\left(X_{i} - E(X_{i}) \right)\left(X_{j} - E(X_{j}) \right)\right] = E\left[\left(X_{i} - \mu_{i} \right)\left(X_{j} - \mu_{j} \right)\right]
$$

Note that If $X_{1}, X_{2}, \ldots , X_{n}$ are independent, then the covariances are $0$ and the covariance matrix is equal to $diag \left(Var(X_{1}), Var(X_{2}), \ldots , Var(X_{n})\right)$ if the $X_{i}$ have common variance $\sigma^{2}$.

__Properties:__

* **Addition to the constant vectors**: Let a be a constant $n \times 1$ vector and let $X$ be a $n \times 1$ random vector. Then, $cov(a + \mathbf{X}) = cov(\mathbf{X})$.

* **Multiplication by constant matrices**: Let $b$ be a constant $m \times n$ matrix and let $X$ be a $n \times 1$ random vector. Then, $cov(b \mathbf{X}) = b cov(\mathbf{X}) b^{T}$.

* **Linear transformations** Let a be a constant $n \times 1$ vector, and $b$ be a constant $m \times n$ matrix and $X$ be a $n \times 1$ random vector. Then, combining the two properties above, one obtains $cov(a + b \mathbf{X})= b cov(\mathbf{X}) b^{T}$.

* **Symmetry**: The covariance matrix $cov(\mathbf{X})$ is a symmetric matrix, that is, it is equal to its transpose: $cov(\mathbf{X}) = cov(\mathbf{X})^{T}$.

* **Positive semi-definiteness**: The covariance matrix $cov(\mathbf{X})$ is positive semi-definite, that is, for a constant $n \times 1$ vector, 
  
  
  $$
a^{T} cov(\mathbf{X}) a \geq 0
$$
  
  This is easily proved,
  
  $$
  \begin{split}
  a^{T} cov(\mathbf{X}) a &= a^{T} E\left[ \left( \mathbf{X} - E(\mathbf{X}) \right) \left( \mathbf{X} - E(\mathbf{X}) \right)^{T} \right] a \\
  &=E\left[a^{T} \left( \mathbf{X} - E(\mathbf{X}) \right) \left( \mathbf{X} - E(\mathbf{X}) \right)^{T} a\right]\\
  &=E\left[\left(\left( \mathbf{X} - E(\mathbf{X}) \right)^{T}a\right)^{T} \left(\left( \mathbf{X} - E(\mathbf{X}) \right)^{T}a\right)\right]\\
  &= E\left[\left(\left( \mathbf{X} - E(\mathbf{X}) \right)^{T}a\right)^{2}\right] \geq 0
  \end{split}
  $$

#### What is Cross-covariance?

Let $\mathbf{X}$ be a $K \times 1$ random vector and $\mathbf{Y}$ be a $L	\times 1$ random vector. The covariance matrix between $\mathbf{X}$ and $ \mathbf{Y}$, or cross-covariance between $\mathbf{X}$ and $\mathbf{Y}$ is denoted by $cov(\mathbf{X}, \mathbf{Y})$. It is defined as follows:

$$
cov(\mathbf{X}, \mathbf{Y}) = E \left[\left(\mathbf{X}-E[\mathbf{X}]\right)\left(\mathbf{Y}-E[\mathbf{Y}]\right)^{T}\right]
$$

provided the above expected values exist and are well-defined.

It is a multivariate generalization of the definition of covariance between two scalar random variables.

Let $X_{1}, \ldots, X_{K}$ denote the $K$ components of the vector $\mathbf{X}$ and $Y_{1}, \ldots, Y_{L}$ denote the $L$ components of the vector $\mathbf{Y}$ .

$$
\begin{split}
cov(\mathbf{X}, \mathbf{Y}) &= 
\begin{bmatrix} E \left[\left(X_{1}-E[X_{1}]\right)\left(Y_{1}-E[Y_{1}]\right)\right] & \ldots & E \left[\left(X_{1}-E[X_{1}]\right)\left(Y_{L}-E[Y_{L}]\right)\right] \\
E \left[\left(X_{2}-E[X_{2}]\right)\left(Y_{1}-E[Y_{1}]\right)\right] & \ldots & E \left[\left(X_{2}-E[X_{2}]\right)\left(Y_{L}-E[Y_{L}]\right)\right] \\
\vdots & \ddots & \vdots \\
E \left[\left(X_{K}-E[X_{K}]\right)\left(Y_{1}-E[Y_{1}]\right)\right] & \ldots & E \left[\left(X_{K}-E[X_{K}]\right)\left(Y_{L}-E[Y_{L}]\right)\right] \\
\end{bmatrix} \\
&=
\begin{bmatrix} cov (X_{1}, Y_{1}) & \cdots & cov (X_{1}, Y_{L})\\
cov (X_{2}, Y_{1}) & \cdots & cov (X_{2}, Y_{L})\\
\vdots & \ddots & \vdots \\
cov (X_{K}, Y_{1}) & \cdots & cov (X_{K}, Y_{L})\\
\end{bmatrix} 
\end{split}
$$

Note that $cov(\mathbf{X}, \mathbf{Y})$ is not the same as $cov(\mathbf{Y}, \mathbf{X})$. In fact, $cov(\mathbf{Y}, \mathbf{X})$ is a $L \times K$ matrix equal to the transpose of $cov(\mathbf{X}, \mathbf{Y})$:

$$
\begin{split}
cov(\mathbf{Y}, \mathbf{X}) &= E \left[\left(\mathbf{Y}-E[\mathbf{Y}]\right)\left(\mathbf{X}-E[\mathbf{X}]\right)^{T}\right]\\
& = E \left[\left(\mathbf{Y}-E[\mathbf{Y}]\right)\left(\mathbf{X}-E[\mathbf{X}]\right)^{T}\right]^{T}\\
&= E \left[\left(\mathbf{X}-E[\mathbf{X}]\right)\left(\mathbf{Y}-E[\mathbf{Y}]\right)^{T}\right] \\
&= cov(\mathbf{X}, \mathbf{Y})
\end{split}
$$

by using the fact that $\left(A B \right)^{T} = B^{T} A^{T}$.

#### What is the correlation for a vector of random variables? How is it related to covariance matrix?

The correlation matrix of $\mathbf{X}$ is defined as

$$
corr(\mathbf{X}) = corr(X_{i}, X_{j}) = 
\begin{bmatrix} 1 & corr(X_{1},X_{2}) & \ldots & corr(X_{1},X_{n}) \\
corr(X_{2}, X_{1}) & 1 & \ldots & corr(X_{2},X_{n}) \\
\vdots & \vdots & \ddots & \vdots \\
corr(X_{n}, X_{1}) & corr(X_{n}, X_{2}) & \ldots & 1 \\
\end{bmatrix}
$$

Denote $cov(\mathbf{X})$ by $\Sigma = (\sigma_{ij})$. Then the correlation matrix and covariance matrix are related by

$$
cov(\mathbf{X}) = diag\left(\sqrt{\sigma_{11}},\sqrt{\sigma_{22}}, \ldots,\sqrt{\sigma_{nn}}\right) \times corr(\mathbf{X}) \times diag\left(\sqrt{\sigma_{11}},\sqrt{\sigma_{22}}, \ldots,\sqrt{\sigma_{nn}}\right)
$$

This is easily seen using $corr(X_{i}, X_{j}) = \dfrac{cov(X_{i}, X_{j})}{\sqrt{\sigma_{ii}\sigma_{jj}}}$

Do not forget that covariance indicates the direction of the linear relationship between variables. Correlation on the other hand measures both the strength and direction of the linear relationship between two variables. Correlation is a function of the covariance. 

Note that covariance and correlation are the same if the features are standardized, i.e., they have mean 0 and variance 1.

#### What are the properties of Distributions?

* **Measures of Central Tendancy**

  - The mean is measured by taking the sum divided by the number of observations.
  - The median is the middle observation in a series of numbers. If the number of observations are even, then the two middle observations would be divided by two.
  - The mode refers to the most frequent observation.
  - The main question of interest is whether the sample mean, median, or mode provides the most accurate estimate of central tendancy within the population.

* **Measures of Dispersion**

  - The standard deviation of a set of observations is the square root of the average of the squared deviations from the mean. The squared deviations from the mean is called the variance.

* **The Shape of Distributions**

  - Unimodal distributions have only one peak while multimodal distributions have several peaks.
  - An observation that is skewed to the right contains a few large values which results in a long tail towards the right hand side of the chart.
  - An observation that is skewed to the left contains a few small values which results in a long tail towards the left hand side of the chart.
  - The kurtosis of a distribution refers to the degree of peakedness of a distribution.
  
  ![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/pearson-mode-skewness.jpg?raw=true)

#### What are the measures of Central Tendency: Mean, Median, and Mode?

The central tendency of a distribution represents one characteristic of a distribution. In statistics, the three most common measures of central tendency are the mean, median, and mode. Each of these measures calculates the location of the central point using a different method. The median and mean can only have one value for a given data set. The mode can have more than one value.

Choosing the best measure of central tendency depends on the type of data you have.

* **Mean**: The mean is the arithmetic average. Calculating the mean is very simple. You just add up all of the values and divide by the number of observations in your dataset.

  $$
\bar{x} = \frac{x_{1}+x_{2}+\cdots +x_{n}}{n}
$$

  The calculation of the mean incorporates all values in the data. If you change any value, the mean changes. However, the mean doesn't always locate the center of the data accurately. In a symmetric distribution, the mean locates the center accurately. However, in a skewed distribution, the mean can miss the mark. Outliers have a substantial impact on the mean. Extreme values in an extended tail pull the mean away from the center. As the distribution becomes more skewed, the mean is drawn further away from the center. Consequently, it’s best to use the mean as a measure of the central tendency when you have a symmetric distribution.

* **Median**: The median is the middle value. It is the value that splits the dataset in half. To find the median, order your data from smallest to largest, and then find the data point that has an equal amount of values above it and below it. The method for locating the median varies slightly depending on whether your dataset has an even or odd number of values.

  If the number of observations is odd, the number in the middle of the list is the median. This can be found by taking the value of the $(n+1)/2$-th term, where n is the number of observations. Else, if the number of observations is even, then the median is the simple average of the middle two numbers. 

  In a symmetric distribution, the mean and median both find the center accurately. They are approximately equal.

  Outliers and skewed data have a smaller effect on the median. Unlike the mean, the median value doesn’t depend on all the values in the dataset. When you have a skewed distribution, the median is a better measure of central tendency than the mean.

  You can also use median when you have ordinal data.
  
* **Mode**: The mode is the value that occurs the most frequently in your data set. Typically, you use the mode with nominal (categorical), ordinal, and discrete (count) data. In fact, the mode is the only measure of central tendency that you can use with norminal (categorical) data. However, with nominal (categorical) data, there is not a central value because you can not order the groups. With ordinal and discrete (count) data, the mode can be a value that is not in the center. Again, the mode represents the most common value.

  In the continuous data, no values repeat, which means there is no mode. With continuous data, it is unlikely that two or more values will be exactly equal because there are an infinite number of values between any two values. When you are working with the raw continuous data, don’t be surprised if there is no mode. However, you can find the mode for continuous data by locating the maximum value on a probability distribution plot. If you can identify a probability distribution that fits your data, find the peak value and use it as the mode.

* When you have a symmetrical distribution for continuous data, the mean, median, and mode are equal.
* When to use the mean: Symmetric distribution, Continuous data
* When to use the median: Skewed distribution, Continuous data, Ordinal data
* When to use the mode: Categorical data, Ordinal data, Count data, Probability Distributions

#### What are the properties of an estimator?

Let $\theta$ be a population parameter. Let $\hat{\theta}$ a sample estimate of that parameter. Desirable properties of $\hat{\theta}$ are: 

* **Unbiased**: A statistic (estimator) is said to be an unbiased estimate of a given parameter when the mean of the sampling distribution of that statistic can be shown to be equal to the parameter being estimated, that is, $E(\hat{\theta}) = \theta$. For example, $E(\bar{X}) = \mu$ and $E(s^{2}) = \sigma^{2}$.

* **Efficiency**: The most efficient estimator among a group of unbiased estimators is the one with the smallest variance. For example, both the sample mean and the sample median are unbiased estimators of the mean of a normally distributed variable. However, $\bar{X}$ has the smallest variance.

* **Sufficiency**: An estimator is said to be sufficient if it uses all the information about the population parameter that the sample can provide. The sample median is not sufficient, because it only uses information about the ranking of observations. The sample mean is sufficient. 

* **Consistency**: An estimator is said to be consistent if it yields estimates that converge in probability to the population parameter being estimated as $N$ becomes larger. That is, as $N$ tends to infinity, $E(\hat{\theta}) = \theta$ , $V(\hat{\theta}) = 0$. For example, as $N$ goess to infinity, $V(\bar{X}) = \frac{\sigma^{2}}{N} = 0$. 

#### What is a Bernoulli distribution? Calculate the expectation and variance of a random variable that follows Bernoulli distribution?
Suppose you perform an experiment with two possible outcomes: either success or failure. Success happens with probability $p$ while failure happens with probability $1-p$. A random variable that takes value $1$ in case of success and $0$ in case of failure is called a Bernoulli random variable.

$X$ has Bernoulli distribution with parameter $p$, the shorthand $X \sim Bernoulli(p), 0 \leq p \leq 1$, its probability mass function is given by:

$$
P_{X}(x) = \left\{ \begin{array}{ll}
         p & \mbox{if $x = 1 $};\\
        1-p & \mbox{if $x  = 0 $}.\end{array} \right.
$$

This can also be expressed as:

$$
P_{X}(x) = p^{x} (1-p)^{1-x},\,\,\, x \in \{0, 1 \}\,\,\,\text{for}\,\,\, 0 \leq p \leq 1
$$

Bernoulli distribution is a special case of Binomial distribution. If $X_{1},\dots ,X_{n}$ are independent, identically distributed (i.i.d.) random variables, all Bernoulli trials with success probability $p$, then their sum is distributed according to a Binomial distribution with parameters $n$ and $p$:

$$
\sum_{k=1}^{n} X_{k} \sim Binomial(n,p)
$$

The Bernoulli distribution is simply $Binomial(1,p)$, also written as $Bernoulli(p)$.

Its expected value is:

$$
E(X) = \sum x p(x) = 1 \times p + 0 \times (1-p) = p
$$

Its variance is:

$$
Var(X) = E(X^{2}) - \left(E(X) \right)^{2} = \sum x^{2} p(x) - p^{2} = 1^{2} \times p + 0^{2} (1-p) - p^{2} = p - p^{2} = p (1-p)
$$

Distribution function of a Bernoulli random variable is:

$$
F_{X}(x) = P(X \leq x) = \left\{ \begin{array}{ll}
         0 & \mbox{if $x < 0 $};\\
        1-p & \mbox{if $0 \leq x < 1 $};\\
        1 & \mbox{if $0 \geq 1 $}.\end{array} \right.
$$

and the fact that $X$ can take either value $0$ or value $1$. If $x<0$, then $P(X \leq x) = 0$ because $X$ can not take values strictly smaller than $0$. If $0 \leq x < 1$, then $P(X \leq x) = 1-p$ because $0$ is the only value strictly smaller than 1 that $X$ can take. Finally, if $x \geq 1$, then $P(X \leq x) = 1$ because all values $X$ can take are smaller than or equal to $1$.

Finding maximum likelihood estimation of the parameter $p$ of Bernoulli distribution is trivial. 

$$
L(p;x) = \prod\limits_{i=1}^n p(x_i;p) = \prod\limits_{i=1}^n p^{x_{i}}(1-p)^{1-x_{i}}
$$

Differentiating the log of $L(p;x)$ with respect to $p$ and setting the derivative to zero shows that this function achieves a maximum at $\hat{p} = \frac{\sum_{i=1}^{n} x_{i}}{n}$.

#### What is Binomial distribution?

A Binomial distribution can be thought of as simply the probability of success or failure outcome in an experiment or a survery that is repeated multiple times. The Binomial distribution is a type of distribution that has two possible outcomes (the prefix 'bi' means two or twice). For example, a coin toss has only two possible outcomes: heads or tails.

Note that, each trial is independent. In other words, none of the trials (experiments) have an effect on the probability of the next trials.

Binomial distribution is probably the most commonly used discrete distribution. 

Suppose that $x = (x_{1}, x_{2}, \ldots, x_{n})$ represents the outcomes of $n$ independent Bernoulli trials, each with success probability $p$, then, its probabiliy mass function is given by:

$$
Binomial(x; n,p) = ^nC_{x} p^{x} (1-p)^{n-x} =  {n \choose x} p^{x} (1-p)^{n-x} = \frac{n!}{x!(n-x)!} p^{x} (1-p)^{n-x}
$$

__Binomial coefficient__, $^nC_{x}$, stated as "n choose k", is akso known as "the number of possible ways to choose $k$ successes from $n$ observations.

The formula for the Binomial cumulative probability function is:

$$
F(x; n, p) = P(X \leq x) = \sum_{i=1}^{x} p^{i} (1-p)^{n-i}
$$

Its mean is $E(x) = np$ and its variance is $Var(x) = np(1-p)$. 

Finding maximum likelihood estimation of the parameter $p$ of Binomial distribution is straightforward. Since we know the probability mass function of Binomial distribution, we can write its likelihood function (joint distribution, since Since $x_{1}, x_{2}, \ldots, x_{n}$ are iid random variables):

$$
\begin{split}
L(p) &= \prod_{i=1}^{n} p(x_{i}; n, p)\\
&=\prod_{i=1}^{n} \frac{n!}{x_{i}!(n-x_{i})!} p^{x_{i}} (1-p)^{n-x_{i}}\\
&=p^{\sum_{i=1}^{n} x_{i}} (1-p)^{\left(n-\sum_{i=1}^{n} x_{i}\right)} \left(\prod_{i=1}^{n} \frac{n!}{x_{i}!(n-x_{i})!}  \right)
\end{split}
$$

Since the last term does not depend on the parameter $p$, we consider it as a constant. We can omit the constant, because it is statistically irrelevant.

In practice, it is more convenient to maximize the log of the likelihood function. Because the logarithm is monotonically increasing function of its argument, maximization of the log of a function is equivalent to maximization of the function itself. Taking the log not only simplifies the subsequent mathematical analysis, but it also helps numerically because the product of a large number of small probabilities can easily underflow the numerical precision of the computer, and this is resolved by computing instead the sum of the log probabilities. Therefore, let's take the log of this likelihood function:

$$
ln L(p) = ln(p)\sum_{i=1}^{n} x_{i} + ln(1-p) \left(n - \sum_{i=1}^{n} x_{i} \right)
$$

In order to find maximum likelihood estimator, we need to take first-order derivative of this function with respect to $p$ and set it to zero:

$$
\frac{\partial}{\partial p} ln L(p) = \frac{1}{p}\sum_{i=1}^{n} x_{i} + \frac{1}{1-p} \left(n - \sum_{i=1}^{n} x_{i} \right) = 0
$$

Solving this equation will yield:

$$
\hat{p} = \frac{\sum_{i=1}^{n} x_{i}}{n}
$$

The numerator $\sum_{i=1}^{n} x_{i}$ is the total number of successes observed in $n$ independent trials. $\hat{p}$ is the observed proportion of successes in the $n$ trials. We often call $\hat{p}$ the sample proportion to distinguish it from $p$, the "true" or "population" proportion.

The fact that the MLE based on n independent Bernoulli random variables and the MLE based on a single binomial random variable are the same is not surprising, since the binomial is the result of $n$ independent Bernoulli trials anyway. In general, whenever we have repeated, independent Bernoulli trials with the same probability of success $p$ for each trial, the MLE will always be the sample proportion of successes. This is true regardless of whether we know the outcomes of the individual trials or just the total number of successes for all trials.

#### What is a multinoulli distribution?

In probability theory and statistics, a categorical distribution (also called a generalized Bernoulli distribution, multinoulli distribution) is a discrete probability distribution that describes the possible results of a random variable that can take on one of $k$ possible categories, with the probability of each category separately specified. The categorical distribution is the generalization of the Bernoulli distribution for a categorical random variable, i.e. for a discrete variable with more than two possible outcomes, such as the roll of a die. A single roll of a die that will have an outcome in $\\{1, 2, 3, 4, 5, 6\\}$, e.g. $k=6$. In the case of a single roll of a die, the probabilities for each value would be $1/6$, or about $0.166$ or about $16.6\%$. On the other hand, the categorical distribution is a special case of the multinomial distribution, in that it gives the probabilities of potential outcomes of a single drawing rather than multiple drawings, therefore, $n=1$. 

A common example of a Multinoulli distribution in machine learning might be a multi-class classification of a single example into one of $K$ classes, e.g. one of three different species of the iris flower.

There is only one trial which produces $k \geq 2$ possible outcomes, with the probabilities, $\pi_{1}, \pi_{2}, \ldots ,\pi_{k}$, respectively. If $X$ is a multinoulli random variables, it can take $x_{1}, x_{2}, \ldots , x_{k}$ different values/outcomes, each with different probabilities (In Bernoulli case, a random variables $X$ can only take two values: either success (1) with probability $p$ or failure(0) with probability $1-p$).

$$
\pi_1+\pi_2+\ldots+\pi_k = 1,\,\,\,\,\, 0 \leq \pi_{j} \leq 1\text{ for } j=1,2, \ldots k
$$

Therefore, probability mass function is given by:

$$
P_{X} (x_{1}, x_{2}, \ldots , x_{k})  = \prod_{i=1}^{k} \pi_{i}^{x_{i}}
$$

where $\sum_{i=1}^{k} x_{i} = 1$.

If you are puzzled by the above definition of the joint pmf, note that when $x_{i}=1$ because the i-th outcome has been obtained, then all other entries are equal to 0 and

$$
\begin{split}
\prod_{i=1}^{k} \pi_{i}^{x_{i}} &= \pi_{1}^{x_{1}} \pi_{2}^{x_{2}} \ldots \pi_{i-1}^{x_{i-1}} \pi_{i}^{x_{i}} \pi_{i+1}^{x_{i+1}} \ldots \pi_{k}^{x_{k}}\\
&=\pi_{1}^{0} \pi_{2}^{0} \ldots \pi_{i-1}^{0} \pi_{i}^{1} \pi_{i+1}^{0} \ldots \pi_{k}^{0}\\
&= \pi_{i}
\end{split}
$$

When $\pi_{i} = \frac{1}{k}$ we get the uniform distribution, which is a special case.

Note that a sum of independent Multinoulli random variables is a multinomial random variable. 

#### What is a multinomial distribution?

Multinomial distribution is a generalization of Binomial distribution, where each trial has $k \geq 2$ possible outcomes.

Suppose that we have an experiment with

* $n$ independent trials, where
* each trial produces exactly one of the events $E_{1}, E_{2}, \ldots, E_{k}$ (i.e. these events are mutually exclusive and collectively exhaustive), and
* on each trial, $E_{j}$ occurs with probability $\pi_{j}$, $j = 1, 2, \ldots, k$.

Notice that $\pi_{1} + \pi_{2} + \ldots + \pi_{k} = 1$. The probabilities, regardless of how many possible outcomes, will always sum to 1.

Here, random variables are:

$X_{1} = \text{ number of trials in which }E_{1}\text{ occurs}$,

$X_{2} = \text{ number of trials in which }E_{2}\text{ occurs}$

...

$X_{k} = \text{ number of trials in which }E_{k}\text{ occurs}$.

Then $X = (X_{1}, X_{2}, \ldots, X_{k})$ is said to have a multinomial distribution with index $n$ and parameter $\pi = (\pi_{1}, \pi_{2}, \ldots , \pi_{k})$. In most problems, $n$ is regarded as fixed and known.

The individual components of a multinomial random vector are binomial and have a binomial distribution,

$X_{1} \sim Bin(n, \pi_{1})$,

$X_{2} \sim Bin(n, \pi_{2})$,

...

$X_{k} \sim Bin(n, \pi_{k})$.

The trials or each person's responses are independent, however, the components or the groups of these responses are not independent from each other. The sample sizes are different now and known. The number of responses for one can be determined from the others. In other words, even though the individual $X_{j}$'s are random, their sum:

$$
X_{1} + X_{2} + \ldots + X_{k} = n
$$

is fixed. Therefore, the $X_{j}$'s are negatively correlated.

If $X = (X_{1}, X_{2}, \ldots, X_{k})$ is multinomially distributed with index $n$ and parameter $\pi = (\pi_{1}, \pi_{2}, \ldots , \pi_{k})$, then we will write $X \sim Mult($n$, \pi)$.

The probability that $X = (X_{1}, X_{2}, \ldots, X_{k})$ takes a particular value $x = (x_{1}, x_{2}, \ldots, x_{k})$ is

$$
P(X_{1} = x_{1}, X_{2} = x_{2}, \ldots, X_{k} =x_{k}  \mid n, \pi_{1}, \pi_{2}, \ldots , \pi_{k}) =\dfrac{n!}{x_1!x_2!\cdots x_k!}\pi_1^{x_1} \pi_2^{x_2} \cdots \pi_k^{x_k}
$$

The possible values of $X$ are the set of $x$-vectors such that each $x_{j} \in \\{0, 1, \ldots , n\\}$ and $x_{1} + \ldots + x_{k} = n$.

If $X \sim Mult(n, \pi)$ and we observe $X = x$, then the loglikelihood function for $\pi$ is:

$$
l(\pi;x)=x_1 \text{log}\pi_1+x_2 \text{log}\pi_2+\cdots+x_k \text{log}\pi_k
$$

Using multivariate calculus, it's easy to maximize this function subject to the constraint

$$
\pi_1+\pi_2+\ldots+\pi_k = 1
$$

the maximum is achieved at

$$
\begin{split}
p &= n^{-1}x\\ &= (x_1/n,x_2/n,\ldots,x_k/n)
\end{split}
$$

the vector of sample proportions. The ML estimate for any individual $\pi_{j}$ is $p_{j} = \dfrac{x_{j}}{n}$, and an approximate $95\%$ confidence interval for $\pi_{j}$ is

$$
p_j \pm 1.96 \sqrt{\dfrac{p_j(1-p_j)}{n}}
$$

because $X_{j} \sim Bin(n, \pi_{j})$. Therefore, the expected number of times the outcome $i$ was observed over $n$ trials is

$$
E(X_{i}) = n \pi_{i}
$$

The covariance matrix is as follows. Each diagonal entry is the variance of a binomially distributed random variable, and is therefore

$$
Var(X_{i})=n \pi_{i}(1-\pi_{i})
$$

The off-diagonal entries are the covariances:

$$
Cov(X_{i}, X_{j})= -n \pi_{i} \pi_{j}
$$

for $i, j$ distinct.

Note that when $k$ is 2 and $n$ is 1, the multinomial distribution is the _Bernoulli distribution_. When $k$ is 2 and $n$ is bigger than 1, it is the _binomial distribution_. When $k$ is bigger than 2 and $n$ is 1, it is the _categorical distribution (multinoulli distribution)_. The Bernoulli distribution models the outcome of a single Bernoulli trial. In other words, it models whether flipping a (possibly biased) coin one time will result in either a success (obtaining a head) or failure (obtaining a tail). The binomial distribution generalizes this to the number of heads from performing n independent flips (Bernoulli trials) of the same coin. The multinomial distribution models the outcome of $n$ experiments, where the outcome of each trial has a categorical distribution, such as rolling a $k$-sided die $n$ times.

**Example**: roll a fair die five times. Here $n = 5$, $k = 6$, and $\pi_{i} = \dfrac{1}{6}$. Our vector $x$ might look like
this:

$$
\begin{bmatrix}
1\\2\\0\\2\\0
\end{bmatrix}
$$

Then $p = \dfrac{5!}{1!2!2!} \dfrac{1}{6}^{1} \dfrac{1}{6}^{2} \dfrac{1}{6}^{2}$.


#### What is the central limit theorem?

The gist of Central Limit Theorem is that the sample mean will be approximately normally distributed for large sample sizes, regardless of the distribution from which we are sampling. 

Suppose we are sampling from a population with mean $\mu$ and standard deviation $\sigma$. Let $\bar{X}$ be a random variable representing the sample mean of $n$ independently drawn observations.

Assuming that $X_{i}$'s are independent and identically distributed, we know that:

* The mean of sampling distribution of the sample mean $\bar{X}$ is equal to the population mean.

  $$
  \mu_{\bar{X}} = E(\bar{X}) = E\left(\frac{X_{1} + X_{2} + \ldots + X_{n}}{n} \right) = \frac{\mu + \mu + \ldots + \mu}{n} = \frac{n\mu}{n} = \mu
  $$

* Standard deviation of the sampling distribution of the sample mean $\bar{X}$ is equal to $\frac{\sigma}{\sqrt{n}}$.

  $$
  Var(\bar{X}) = Var \left(\frac{X_{1} + X_{2} + \ldots + X_{n}}{n} \right) = \frac{1}{n^{2}} \left(\sigma^{2} + \sigma^{2}+ \ldots + \sigma^{2} \right) = \frac{\sigma^{2}}{n}
  $$
  
  which is called the "standard error of the mean".

Given any random variable $X$, discrete or continuous, with finite mean $\mu$ and finite $\sigma^{2}$. Then, regardless of the shape of the population distribution of $X$, as the sample size $n$ gets larger ($n \geq 30$), the sampling distribution of $\bar{X}$ becomes increasingly closer to normal with mean $\mu$ and variance $\frac{\sigma^{2}}{n}$, that is $\bar{X} \sim N\left(\mu ,  \frac{\sigma^{2}}{n}\right)$ approximately. 

More formally,

$$
Z = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \sim N(0,1) \text{ as } n \to \infty
$$
  
What is the population standard deviation $\sigma$ is unknown? Then, it can be replaced by the sample standard deviation $s$, provided $n$ is large that is $\bar{X} \sim N\left(\mu ,  \frac{s^{2}}{n}\right)$. The standard deviation of $\bar{X}$ is referred to as the true standard error of the mean. Since the value $s/\sqrt{n}$ is a sample-based estimate of the true standard error (s.e.), it is commonly denoted as $\hat{s.e.}$. 
  
Sample variance $s^{2}$ is an unbiased estimator of the population variance $\sigma^{2}$ that is $E(s^{2})=\sigma^{2}$.

$$
s^{2} = \frac{1}{n-1} \sum_{i=1}^{n} \left(X_{i} - \bar{X} \right)^{2}
$$

The denominator $n-1$ in the sample variance is necessary to ensure unbiasedness of the population variance.

#### What is the sampling distribution of sample proportion, p-hat?

The Central Limit Theorem has an analogue for the population proportion $\hat{p}$. For large samples, the sample proportion is approximately normally distributed, with mean $\mu_{\hat{p}} = p$ and variance $\sigma_{\hat{p}}^{2} = \sqrt{\frac{p(1-p)}{n}}$, which is also called standard error of $p$. 

In actual practice $p$ is not known, hence neither is $\sigma_{\hat{p}}^{2}$. In that case in order to check that the sample is sufficiently large we substitute the known quantity $\hat{p}$ for $p$. 

#### What is population mean and sample mean?

A population is a collection of persons, objects or items of interest. Population mean is the average of the all the elements in the population. Suppose that whole population consists of $N$ observations:

$$
\mu = \frac{1}{N} \sum_{i=1}^{N} X_{i}
$$

A sample is a portion of the whole and, if properly taken, is representative of the population. Sample mean is the arithmetic mean of random sample values drawn from the population. Let's assume that we take $n$ samples from this population:

$$
\bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_{i}
$$

#### What is population standard deviation and sample standard deviation?

Standard deviation measures the spread of a data distribution around the mean of this distribution. It measures the typical distance between each data point and the mean. Standard deviation is the square root of the variance. 

Population standard deviation ($\sigma$):

$$
\sigma^{2} = \frac{\sum_{i=1}^{N} X_{i} - \mu}{N}
$$

Sample standard deviation ($s$):

$$
s^{2} = \frac{\sum_{i=1}^{n} X_{i} - \bar{X}}{n-1}
$$

#### Why population standard deviation has N degrees of freedom while sample standard deviation has n-1 degrees of freedom? In other words, why 1/N inside root for population and 1/(n-1) inside root for sample standard deviation?

When we calculate the sample standard deviation from a sample of $n$ values, we are using the sample mean already calculated from that same sample of $n$ values.  The calculated sample mean has already "used up" one of the "degrees of freedom of variability" that is available in the sample.  Only $n-1$ degrees of freedom of variability are left for the calculation of the sample standard deviation.

Here's another way to look at it: Suppose someone else draws a random sample of, say, $10$ values from a population. They tell you what $9$ of the $10$ sample values are, and they also tell you the sample mean of the $10$ values. From this information, even though they haven't told you the tenth value, you can now calculate it for yourself. Given the nine sample values and the sample mean, the tenth sample value cannot vary:  it is totally predetermined. The tenth value is not free to vary. Essentially, only nine of the ten values are useful for determining the variability of the sample.  In other words, we would need to use $n-1$ as the degrees of freedom for the variability in the sample.

Statistically, it also comes from the fact that $s^{2}$ is the unbiased estimator of $\sigma^{2}$. in statistics using an unbiased estimator is preferred. For more details, look at [here](https://mmuratarat.github.io/2019-09-27/unbiased-estimator-proof).

#### What is the sampling distribution of the sample mean?

The sampling distribution of a population mean is generated by repeated sampling and recording of the means obtained. This forms a distribution of different means, and this distribution has its own mean and variance.

The sample mean follows a normal distribution with mean $\mu$ and variance $\frac{\sigma^{2}}{n}$. This comes from the fact that sum of independent normal random variables. For details, look [here](https://newonlinecourses.science.psu.edu/stat414/node/172/){:target="_blank"} and [here](https://newonlinecourses.science.psu.edu/stat414/node/173/){:target="_blank"}.

Assuming that $X_{i}$'s are independent and identically distributed, we know that:

* The mean of sampling distribution of the sample mean $\bar{X}$ is equal to the population mean.

  $$
  \mu_{\bar{X}} = E(\bar{X}) = E\left(\frac{X_{1} + X_{2} + \ldots + X_{n}}{n} \right) = \frac{\mu + \mu + \ldots + \mu}{n} = \frac{n\mu}{n} = \mu
  $$

* Standard deviation of the sampling distribution of the sample mean $\bar{X}$ is equal to $\frac{\sigma}{\sqrt{n}}$.

  $$
  Var(\bar{X}) = Var \left(\frac{X_{1} + X_{2} + \ldots + X_{n}}{n} \right) = \frac{1}{n^{2}} \left(\sigma^{2} + \sigma^{2}+ \ldots + \sigma^{2} \right) = \frac{\sigma^{2}}{n}
  $$
  
  which is called the "standard error of the mean".

#### What is the sampling distribution of the sample variance?

If $X_{1}, X_{2}, \ldots , X_{n}$ are iid $N(\mu, \sigma^{2})$ random variables, then,

$$
\frac{n-1}{\sigma^{2}}s^{2} \sim \chi_{n-1}^{2}
$$

The proof is given below from this [Stackexchange.com link](https://math.stackexchange.com/a/47013/45210){:target="_blank"}.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/proof_sampling_dist_sample_variance.png?raw=true)

For the chi-square distribution, mean and the variance are

$$
E(\chi_{\nu}^{2}) = \nu
$$

and 

$$
Var(\chi_{\nu}^{2}) = 2\nu
$$

we can use this to get the mean and variance of $s^{2}$.

$$
E(s^{2}) = E\left(\frac{\sigma^{2} \chi_{n-1}^{2}}{n-1} \right) = \frac{\sigma^{2}}{n-1}n-1 = \sigma^{2}
$$

and similarly,

$$
Var(s^{2}) = Var \left(\frac{\sigma^{2} \chi_{n-1}^{2}}{n-1} \right) =  \frac{\sigma^{4}}{(n-1)^{2}} 2(n-1) = \frac{2\sigma^{4}}{n-1}
$$

#### What is the standard error of the estimate? 

The standard deviation (SD) measures the amount of variability, or dispersion, for a subject set of data from the mean, while the standard error of the mean (SEM) measures how far the sample mean of the data is likely to be from the true population mean. For a given sample size, the standard error equals the standard deviation divided by the square root of the sample size. The SEM is always smaller than the SD. SEM is the SD of the theoretical distribution of the sample means (the sampling distribution).

standard deviation: $s = \sqrt{\frac{\sum_{i=1}^{n} X_{i} - \bar{X}}{n-1}}$

Variance = $s^{2}$

standard error: $s_{\bar{X}} = \sqrt{\frac{\sigma^{2}}{n}}$

where $n$ is the size of the sample and $\bar{X}$ is the sample mean.

#### What is confidence interval?

The purpose of taking a random sample from a population and computing a statistic, such as the mean from the data, is to approximate the mean of the population. How well the sample statistic estimates the underlying population value is always an issue. In statistical inference, one wishes to estimate population parameters using observed sample data. A confidence interval gives an estimated range of values which is likely to include an unknown population parameter, the estimated range being calculated from a given set of sample data

Confidence intervals are constructed at a confidence level, such as $95\%$, selected by the user. What does this mean? It means that if the same population is sampled on numerous occasions and interval estimates are made on each occasion, the resulting intervals would bracket the true population parameter in approximately $95\%$ of the cases.

For example, when we try to construct confidence interval for the true mean of heights of men, the "$95\%$" says that $95$ of $100$ experiments will include the true mean, but $5$ won't. So there is a 1-in-20 chance ($5\%$) that our confidence interval does NOT include the true mean. 

In the same way that statistical tests can be one or two-sided, confidence intervals can be one or two-sided. A two-sided confidence interval brackets the population parameter from above and below. A one-sided confidence interval brackets the population parameter either from above or below and furnishes an upper or lower bound to its magnitude.
Confidence intervals only assess sampling error in relation to the parameter of interest. (Sampling error is simply the error inherent when trying to estimate the characteristic of an entire population from a sample.) Consequently, you should be aware of these important considerations:

* As you increase the sample size, the sampling error decreases and the intervals become narrower. If you could increase the sample size to equal the population, there would be no sampling error. In this case, the confidence interval would have a width of zero and be equal to the true population parameter.
* Confidence intervals only tell you about the parameter of interest and nothing about the distribution of individual values.

#### What is a p-value?
Before we talk about what p-value means, let’s begin by understanding hypothesis testing where p-value is used to determine the statistical significance of our results, which is our ultimate goal. 

When you perform a hypothesis test in statistics, a p-value helps you determine the significance of your results. Hypothesis tests are used to test the validity of a claim that is made about a population using sample data. This claim that’s on trial is called the null hypothesis. The alternative hypothesis is the one you would believe if the null hypothesis is concluded to be untrue. It is the opposite of the null hypothesis; in plain language terms this is usually the hypothesis you set out to investigate. 

In other words, we’ll make a claim (null hypothesis) and use a sample data to check if the claim is valid. If the claim isn’t valid, then we’ll choose our alternative hypothesis instead. Simple as that. These two hypotheses specify two statistical models for the process that produced the data. 

To know if a claim is valid or not, we’ll use a p-value to weigh the strength of the evidence to see if it’s statistically significant. If the evidence supports the alternative hypothesis, then we’ll reject the null hypothesis and accept the alternative hypothesis. 

p-value is a measure of the strength of the evidence provided by our sample against the null hypothesis. In other words, p-value is the probability of getting the observed value of the test statistics, or a value with even greater evidence against null hypothesis, if the null hypothesis is true. Smaller the p-value, the greater the evidence against the null hypothesis. 

If we are given a significance level, i.e., alpha, then we reject null hypothesis if p-value is less than equal the chosen significance level, i.e., accept that your sample gives reasonable evidence to support the alternative hypothesis. The term significance level (alpha) is used to refer to a pre-chosen probability and the term "p-value" is used to indicate a probability that you calculate after a given study. The choice of significance level at which you reject null hypothesis is arbitrary. Conventionally the $5\%$ (less than $1$ in $20$ chance of being wrong), $1\%$ and $0.1\%$ ($p < 0.05, 0.01\text{ and }0.001$) levels have been used. Most authors refer to statistically significant as $p < 0.05$ and statistically highly significant as $p < 0.001$ (less than one in a thousand chance of being wrong).

A fixed level alpha test can be calculated without first calculating a p-value. This is done by comparing the test statistic with a critical value of the null distribution corresponding to the level alpha. This is usually the easiest approach when doing hand calculations and using statistical tables, which provide percentiles for a relatively small set of probabilities. Most statistical software produces p-values which can be compared directly with alpha. There is no need to repeat the calculation by hand.

You can use either P values or confidence intervals to determine whether your results are statistically significant. If a hypothesis test produces both, these results will agree.

The confidence level is equivalent to 1 – the alpha level. So, if your significance level is $0.05$, the corresponding confidence level is $95\%$.

* If the P value is less than your significance (alpha) level, the hypothesis test is statistically significant.
* If the confidence interval does not contain the null hypothesis value, the results are statistically significant.
* If the P value is less than alpha, the confidence interval will not contain the null hypothesis value.

#### What do Type I and Type II errors mean?

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/type_I_type_II_errors.png?raw=true)

__Type I error__ is the rejection of a true null hypothesis (also known as a "false positive" finding or conclusion), while a __type II error__ is the non-rejection of a false null hypothesis (also known as a "false negative" finding or conclusion). 

When the null hypothesis is true and you reject it, you make a type I error. The probability of making a type I error is alpha, which is the level of significance you set for your hypothesis test.

The power of a test is one minus the probability of type II error (beta), which is the probability of rejecting the null hypothesis when it is false. In other words, it is the ability to detect a fault when there is actually a fault to be detected. Therefore, power should be maximized when selecting statistical methods. 

The chances of committing these two types of errors are inversely proportional—that is, decreasing Type I error rate increases Type II error rate, and vice versa. To decrease your chance of committing a Type I error, simply make your alpha value more stringent. To reduce your chance of committing a Type II error, increase your analyses’ power by either increasing your sample size or relaxing your alpha level!

In a drug effectiveness study, a false positive could cause the patient to use an ineffective drug. Conversely, a false negative could mean not using a drug that is effective at curing the disease. Both cases could have a very high cost to the patient’s health.

In a machine learning A/B test, a false positive might mean switching to a model that should increase revenue when it doesn’t. A false negative means missing out on a more beneficial model and losing out on potential revenue increase.

A statistical hypothesis test allows you to control the probability of false positives by setting the significance level, and false negatives via the power of the test. If you pick a false positive rate of 0.05, then out of every 20 new models that don’t improve the baseline, on average 1 of them will be falsely identified by the test as an improvement.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/confusion_matrix.png?raw=true)

In the context of confusion matrix, we can say Type I error occurs when we classify a value as positive (1) when it is actually negative (0). For example, false-positive test result indicates that a person has a specific disease or condition when the person actually does not have it. 

Type II error occurs when we classify a value as negative (0) when it is actually positive (1). Similarly, as an example, a false negative is a test result that indicates a person does not have a disease or condition when the person actually does have it


## General Machine Learning

#### What is the matrix used to evaluate the predictive model? How do you evaluate the performance of a regression prediction model vs a classification prediction model?

Confusion Matrix, also known as an error matrix, is a specific table layout that allows visualization of the complete performance of an algorithm. Each row of the matrix represents the instances in a predicted class while each column represents the instances in an actual class (or vice versa). The name stems from the fact that it makes it easy to see if the system (algorithm) is confusing two classes (i.e. commonly mislabeling one as another). It is a special kind of contingency table, with two dimensions ("actual" and "predicted"), and identical sets of "classes" in both dimensions (each combination of dimension and class is a variable in the contingency table).

* **Regression problems**: Mean Absolute Error, Mean Squared Error, Root Mean Squared Error, R-squared
* **Classification problems**: Accuracy, Precision, Recall, Sensitivity, Specificity, False Positive Rate, F1 Score, AUC, Lift and gain charts

#### What are the assumptions required for linear regression?

* Linear Relationship between the features and target
* The number of observations must be greater than number of features
* No Multicollinearity between the features: Multicollinearity is a state of very high inter-correlations or inter-associations among the independent variables.It is therefore a type of disturbance in the data if present weakens the statistical power of the regression model. Pair plots and heatmaps(correlation matrix) can be used for identifying highly correlated features.
* Homoscedasticity of residuals or equal variance $Var \left(\varepsilon \mid X_{1} = x_{1}, \cdots, X_{p}=x_{p} \right) = \sigma^{2}$: Homoscedasticity describes a situation in which the error term (that is, the "noise" or random disturbance in the relationship between the features and the target) is the same across all values of the independent variables. 
     More specifically, it is assumed that the error (a.k.a residual) of a regression model is homoscedastic across all values of the predicted value of the dependent variable. A scatter plot of residual values vs predicted values is a good way to check for homoscedasticity. There should be no clear pattern in the distribution and if there is a specific pattern, the data is heteroscedastic. 
* Normal distribution of error terms $\varepsilon \sim N(0, \sigma^{2})$: The fourth assumption is that the error(residuals) follow a normal distribution.However, a less widely known fact is that, as sample sizes increase, the normality assumption for the residuals is not needed. More precisely, if we consider repeated sampling from our population, for large sample sizes, the distribution (across repeated samples) of the ordinary least squares estimates of the regression coefficients follow a normal distribution. As a consequence, for moderate to large sample sizes, non-normality of residuals should not adversely affect the usual inferential procedures. This result is a consequence of an extremely important result in statistics, known as the central limit theorem.
     Normal distribution of the residuals can be validated by plotting a q-q plot.
* No autocorrelation of residuals (Independence of errors $E(\varepsilon_{i} \varepsilon_{j}] = 0, \,\,\, i \neq j$): Autocorrelation occurs when the residual errors are dependent on each other. The presence of correlation in error terms drastically reduces model's accuracy. This usually occurs in time series models where the next instant is dependent on previous instant.
    Autocorrelation can be tested with the help of Durbin-Watson test. The null hypothesis of the test is that there is no serial correlation. 
    
#### What are the assumptions required for logistic regression?

First, logistic regression does not require a linear relationship between the dependent and independent variables.  Second, the error terms (residuals) do not need to be normally distributed.  Third, homoscedasticity is not required.  Finally, the dependent variable in logistic regression is not measured on an interval or ratio scale.
 
However, some other assumptions still apply.

* __ASSUMPTION OF APPROPRIATE OUTCOME STRUCTURE:__ Binary logistic regression requires the dependent variable to be binary and ordinal logistic regression requires the dependent variable to be ordinal.

* __ASSUMPTION OF OBSERVATION INDEPENDENCE:__ Logistic regression requires the observations to be independent of each other.  In other words, the observations should not come from repeated measurements or matched data.

* __ASSUMPTION OF THE ABSENCE OF MULTICOLLINEARITY:__ Logistic regression requires there to be little or no multicollinearity among the independent variables.  This means that the independent variables should not be too highly correlated with each other.

* __ASSUMPTION OF LINEARITY OF INDEPENDENT VARIABLES AND LOG ODDS:__ Logistic regression assumes linearity of independent variables and log odds.  although this analysis does not require the dependent and independent variables to be related linearly, it requires that the independent variables are linearly related to the log odds.

* __ASSUMPTION OF A LARGE SAMPLE SIZE:__ Logistic regression typically requires a large sample size.

#### Why is logistic regression considered to be linear model?

Logistic regression is a generalized linear model, is of the form:

$$
\text{logit}(P(y=1)) = log\left(\frac{P(y=1)}{1-P(y=1)}\right)=log\left(\frac{P(y=1)}{P(y=0)}\right)=\theta_{0}+\theta_{1}x_{1}+\ldots+\theta_{p}x_{p}
$$

More generally, in a Generalized Linear Model, the mean, $\mu$, of the distribution depends on the independent variables, $x$, through:

$$
E(y) = g(\mu) = \theta_{0}+\theta_{1}x_{1}+\ldots+\theta_{p}x_{p}
$$

where $\mu$ is the expected value of the response given the covariates.

Consequently, its decision boundary is linear. The decision boundary is the set of $x$ such that

$$
\frac{1}{1 + e^{-{X \cdot \theta}}} = 0.5
$$

This is equivalent to

$$
1 = e^{-{X \cdot \theta}}
$$

and, taking the natural log of both sides,

$$
0 = -X \cdot \theta = -\sum\limits_{i=0}^{p} \theta_i x_i = \theta_{0}+\theta_{1}x_{1}+\ldots+\theta_{p}x_{p}
$$

which defines a straight line. So the decision boundary is linear.

#### Why sigmoid function in Logistic Regression?

One of the nice properties of logistic regression is that the sigmoid function outputs the conditional probabilities of the prediction, the class probabilities because the output range of a sigmoid function is between 0 and 1. This transform ensures that probability lies between 0 and 1.

$$
sigmoid (x)=\dfrac{1}{1+e^{-x}}
$$

![](https://raw.githubusercontent.com/mmuratarat/mmuratarat.github.io/master/_posts/images/sigmoid.png)

#### What is Softmax regression and how is it related to Logistic regression?

Softmax Regression (a.k.a. Multinomial Logistic, Maximum Entropy Classifier, or just Multi-class Logistic Regression) is a generalization of logistic regression that we can use for multi-class classification (under the assumption that the classes are mutually exclusive). In contrast, we use the (standard) Logistic Regression model in binary classification tasks.

#### What is collinearity and what to do with it? How to remove multicollinearity?

**Collinearity/Multicollinearity:**
* In multiple regression: when two or more variables are highly correlated or improper use of dummy variables (e.g. failure to exclude one category).
* They provide redundant information
* In case of perfect multicollinearity: $\beta = (X^{T}X)^{-1}X^{T}y $ does not exist, the design matrix is not invertible
* It doesn't affect the model as a whole, doesn't bias results
* The standard errors of the regression coefficients of the affected variables tend to be large
* The test of hypothesis that the coefficient is equal to zero may lead to a failure to reject a false null hypothesis of no effect of the explanatory (Type II error)
* Leads to overfitting

**Remove multicollinearity:**
* Drop some of affected variables
* Combine the affected variables
* Removing correlated variables might lead to loss of information. In order to retain those variables, we can use penalized regression models like ridge or lasso regression. 
* Partial least square regression
* Principal component regression: gives uncorrelated predictors

**Detection of multicollinearity:**
* Large changes in the individual coefficients when a predictor variable is added or deleted
* Insignificant regression coefficients for the affected predictors but a rejection of the joint hypothesis that those coefficients are all zero (F-test)
* The extent to which a predictor is correlated with the other predictor variables in a linear regression can be quantified as the R-squared statistic of the regression where the predictor of interest is predicted by all the other predictor variables. The variance inflation factor (VIF) for variable $i$ is then computed as:
    
    \begin{equation}
        VIF = \frac{1}{1-R_{i}^{2}}
    \end{equation}
    
    A rule of thumb for interpreting the variance inflation factor: 
    * 1 = not correlated.
    * Between 1 and 10 = moderately correlated.
    * Greater than 10 = highly correlated.
    
     The rule of thumb cut-off value for VIF is 10. Solving backwards, this translates into an R-squared value of 0.90. Hence, whenever the R-squared value between one independent variable and the rest is greater than or equal to 0.90, you will have to face multicollinearity.
     
     Tolerance (1/VIF) is another measure to detect multicollinearity.  A tolerance close to 1 means there is little multicollinearity, whereas a value close to 0 suggests that multicollinearity may be a threat. 

* Correlation matrix. However, unfortunately, multicollinearity does not always show up when considering the variables two at a time. Because correlation is a bivariate relationship whereas multicollinearity is multivariate.
    
* Eigenvalues of the correlation matrix of the independent variables near zero indicate multicollinearity. Instead of looking at the numerical size of the eigenvalue, use the condition number. Large condition numbers indicate multicollinearity.
    
* Investigate the signs of the regression coefficients. Variables whose regression coefficients are opposite in sign from what you would expect may indicate multicollinearity

#### What is R squared?

R-squared ($R^{2}$) is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model. Whereas correlation explains the strength of the relationship between an independent and dependent variable, R-squared explains to what extent the variance of one variable explains the variance of the second variable. So, if the $R^{2}$ of a model is $0.50$, then approximately half of the observed variation can be explained by the model's inputs. It may also be known as the coefficient of determination.

$$
\begin{split}
R^{2} \left(y_{true}, y_{pred} \right) =&  1- \frac{\text{Sum of Squared}_{residuals}}{\text{Sum of Squared}_{total}} 
&= 1 - \frac{\sum \left(y_{true} - y_{pred}\right)^{2}}{\sum \left(y_{true} - \bar{y} \right)^{2}}
\end{split}
$$

where 
$$
\bar{y} = \frac{1}{n_{samples}}\sum {y_{true}}$$

R-squared values range from 0 to 1 and are commonly stated as percentages from $0\%$ to $100\%$. 

R squared alone cannot be used as a meaningful comparison of models with very different numbers of independent variables. It only works as intended in a simple linear regression model with one explanatory variable. R-squared is monotone increasing with the number of variables included—i.e., it will never decrease because when we add a new variable, regression model will try to minimize the sum of squared of residuals but total sum of squared will be the same. Thus, a model with more terms may seem to have a better fit just for the fact that it has more terms. This leads to the alternative approach of looking at the adjusted R squared. The adjusted R-squared compares the descriptive power of regression models that include diverse numbers of predictors. The adjusted R-squared compensates for the addition of variables and only increases if the new term enhances the model above what would be obtained by probability and decreases when a predictor enhances the model less than what is predicted by chance. In an overfitting condition, an incorrectly high value of R-squared, which leads to a decreased ability to predict, is obtained. This is not the case with the adjusted R-squared.

While standard R-squared can be used to compare the goodness of two or model different models, adjusted R-squared is not a good metric for comparing nonlinear models or multiple linear regressions.

$$
\bar{R}^2 = 1- (1- {R^2})\frac{n-1}{n-p-1}
$$

where $p$ is the total number of explanatory variables in the model (not including the constant term), and n is the sample size.

#### You have built a multiple regression model. Your model $R^{2}$ isn't as good as you wanted. For improvement, your remove the intercept term, your model $R^{2}$ becomes 0.8 from 0.3. Is it possible? How?

Yes, it is possible. We need to understand the significance of intercept term in a regression model. The intercept term shows model prediction without any independent variable i.e. mean prediction ($\hat{y}$). The denominator of the formula of $R^{2}$ contains $\hat{y}$.

When intercept term is present, $R^{2}$ value evaluates your model with respect to to the mean model. In absence of intercept term $\hat{y}$, the model can make no such evaluation, with large denominator,

$$
R^2 =1 - \frac{\sum \left(y_{true} - y_{pred} \right)^2}{\sum \left(y_{true} \right)^2}
$$

equation's value becomes smaller than actual, resulting in higher $R^{2}$.

#### How do you validate a machine learning model?

The most important thing you can do to properly evaluate your model is to not train the model on the entire dataset.

* **The train/test/validation split**: A typical train/test split would be to use $70\%$ of the data for training and $30\%$ of the data for testing. It's important to use new data when evaluating our model to prevent the likelihood of overfitting to the training set. However, sometimes it's useful to evaluate our model as we're building it to find that best parameters of a model - but we can't use the test set for this evaluation or else we'll end up selecting the parameters that perform best on the test data but maybe not the parameters that generalize best. To evaluate the model while still building and tuning the model, we create a third subset of the data known as the validation set. A typical train/test/validation split would be to use $60\%$ of the data for training, $20\%$ of the data for validation, and $20\%$ of the data for testing.

* **Random Subsampling (Hold-out) Validation**: This is a simple kind of cross validation technique. You reserve around half of your original dataset for testing(or validation), and the other half for training. Once you get an estimate of the model’s error, you may also use the portion previously used for training for testing now, and vice versa. Effectively, this gives you two estimates of how well your model works. 

* **Leave-One-Out Cross-Validation**: This is the most extreme way to do cross-validation. Assuming that we have $n$ labeled observations, LOOCV trains a model on each possible set of $n-1$ observations, and evaluate the model on the left out one; the error reported is averaged over the $n$ models trained. This technique is computationally very, very intensive- you have to train and test your model as many times as there are number of data points. This can spell trouble if your dataset contains millions of them. 

* **Cross-Validation**: When you do not have a decent validation set to tune the hyperparameters of the model on, the common approach that can help is called cross-validation. In the case of having a few training instances, it could be prohibitive to have both validation and test set separately. You would prefer to use more data to train the model. In such a situation, you only split your data into a training and a test set. Then you can use cross-validation on the training set to simulate a validation set. Cross-validation works as follows. First, you fix the values of hyperparameters you want to evaluate. Then you split your training set into several subsets of the same space. Each subset is called a _fold_. Typically, five-fold or ten-fold provides a good compromise for the bias-variance trade-off. With five-fold CV, you randomly split your training data into five folds: $\{F_{1}, F_{2}, ..., F_{5}\}$. Each $F_{k}$ contains $20\%$ of the training data. Then you train five models as follows. To train the first model, $f_{1}$, you use all examples from folds  $\{F_{2}, F_{3}, F_{4}, F_{5}\}$ as the training set and the examples from $F_{1}$ as the validation set. To train the second model, $f_{2}$, you use the examples from fold $\{F_{1}, F_{3}, F_{4}, F_{5}\}$ to train and the examples from $F_{2}$ as the validation set. You continue building models iteratively like this and compute the value of the metric of interest on each validation sets, from $F_{1}$ to $F_{5}$. Then you average the five values of the metric to get the final value. You can use grid search with cross-validation to find the best values of hyperparameters for your model. Once you have found those values, you use the entire training set to build the model with these best values of parameters you have found via cross-validation. Finally, you assess the model using the test set. 

* **Stratified Cross Validation**:  When we split our data into folds, we want to make sure that each fold is a good representative of the whole data. The most basic example is that we want the same proportion of different classes in each fold. Most of the times it happens by just doing it randomly, but sometimes, in complex datasets, we have to enforce a correct distribution for each fold.

* **Bootstrapping Method**: Under this technique training dataset is randomly selected with replacement and the remaining data sets that were not selected for training are used for testing. The error rate of the model is average of the error rate of each iteration  as estimation of our model performance, the value is likely to change from fold-to-fold during the validation process.

#### What is the Bias-variance trade-off for Leave-one-out and k-fold cross validation?

When we perform Leave-One-Out Cross Validation (LOOCV), we are in effect averaging the outputs of $n$ fitted models (assuming we have $n$ observations), each of which is trained on an almost identical set of observations; therefore, these outputs are highly (positively) correlated with each other. In contrast, when we perform $k$-fold CV with $k < n$ are averaging the outputs of $k$ fitted models that are somewhat less correlated with each other, since the overlap between the training sets in each model is smaller. Since the mean of many highly correlated quantities has higher variance than does the mean of many quantities that are not as highly correlated, the test error estimate resulting from LOOCV tends to have higher variance than does the test error estimate resulting from $k$-fold CV (the variance of the sum of correlated variables increases with the amount of covariance). However, LOOCV estimator is approximately unbiased for the true (expected) prediction error

To summarize, there is a bias-variance trade-off associated with the choice of $k$ in $k$-fold cross-validation.

Note that while two-fold cross validation doesn't have the problem of overlapping training sets, it often also has large variance because the training sets are only half the size of the original sample. A good compromise is ten-fold cross-validation.

Typically, given these considerations, one performs $k$-fold cross-validation with $k=5$ or $k=10,$ as these values have been shown empirically to yield test error rate estimates that suffer neither from excessively high bias nor from very high variance.

#### Describe Machine Learning, Deep Learning, Supervised Learning, Unsupervised Learning, Semi-Supervised Learning, Reinforcement Learning with examples

* **Machine Learning** is the science of getting computers to learn and act like humans do and improve their learning over time in an autonomous fashion, by feeding them data and the information in the form of observations and real-world interactions. It seeks to provide knowledge to computers through data, observations and interacting with the world. That acquired knowledge allows computers to correctly generalize to new settings (can adapt to new data). It is a subset of Artificial Intelligence that uses statistical techniques to give machine the ability to "learn" from data without explicitly given the instructions for how to do so. This process is knows as "training" a "model" using a learning "algorithm" that progressively improves the model performance on a specific task. 

* **Deep Learning** is an area of Machine Learning that attempts to mimic the activity in the layers of neurons in the brain to learn how to recognize the complex patterns in the data. The "deep" in deep learning refers to the large number of layers of neurons in contemporary ML models that help to learn rich representations of data to achieve better performance gains.

* **Supervised Learning**: Supervised learning is where you have input variables (x) and an output variable (Y) and you use an algorithm to learn the mapping function from the input to the output. The goal is to approximate the mapping function given a set of features called predictors so well that when you have new input data (x) that you can predict the output variables (Y) for that data. It is called supervised learning because the process of an algorithm learning from the training dataset can be thought of as a teacher supervising the learning process. We know the correct answers, the algorithm iteratively makes predictions on the training data and is corrected by the teacher. Learning stops when the algorithm achieves an acceptable level of performance. Supervised learning problems can be further grouped into regression and classification problems. Classification (automatically assigning a label to an unlabelled example) is a type problem where the response variable is qualitative (like a category, such as "red" or "blue" or "disease" and "no disease"). Regression (predicting a real-valued label given an unlabelled example) is another type of problem when the output variable (response variable) is a quantitative (real value, integer or floating point number), such as "dollars" or "weight". Some supervised learning algorithms are Support vector machines, neural networks, linear regression, logistic regression, extreme gradient boosting.

* **Unsupervised Learning**: Unsupervised learning is where you only have input data (X) and no corresponding output variables. The goal for unsupervised learning is to model the underlying structure or distribution in the data in order to learn more about the data. These are called unsupervised learning because unlike supervised learning above there is no correct answers and there is no teacher. Algorithms are left to their own devises to discover and present the interesting structure in the data. Unsupervised learning problems can be further grouped into clustering and association problems. Clustering is used to discover the inherent groupings in the data based on some notation of similarity, such as grouping customers by purchasing behavior, where as an association rule learning problem is to discover rules (interesting relations) that describe large portions of your data, such as people that buy X also tend to buy Y. Some unsupervised learning algorithms are principal component analysis, singular value decomposition; identify group of customers

* **Semi-supervised Learning**: In this type of learning, the dataset contains both labeled and unlabeled examples. Usually, the quantity of unlabeled examples is much higher than the number of labeled examples. Many real world machine learning problems fall into this area. This is because it can be expensive or time-consuming to label data as it may require access to domain experts. Whereas unlabeled data is cheap and easy to collect and store. It could look counter-intuitive that learning could benefit from adding more unlabeled examples. It seems like we add more uncertainty to the problem. However, when you add cheap and abundant unlabeled examples, you add more information about the problem. A larger sample reflects better the probability distribution of the data we labeled came from. Most semi-supervised learning algorithms are combinations of unsupervised and supervised algorithms. For example, deep belief networks are based on unsupervised components called restricted Boltzmann machines (RBMs), stacked on the top of another. RBMs are trained sequentially in an unsupervised manner, and then the whole system is fine-tuned using supervised learning techniques, 

* **Reinforcement Learning**: It is a sub-field of machine learning where the machine "lives" in an environment and is capable of perceiving the state of that environment as a vector of features. The machine can also execute actions in every state. Different actions bring different rewards and could also move the machine to another state of the environment. The goal of a reinforcement learning algorithm is to learn a policy (which is the best strategy). In Reinforcement Learning, we want to develop a learning system (called an _agent_) that can learn how to take actions in the real world. The most common approach is to learn those actions by trying to maximize some kind of reward (or minimize penalties in the form of negative rewards) encouraging the desired state of the environment. For example, many robots implement Reinforcement Learning algorithms to learn how to walk. DeepMind's AlphaGo program is also a good example of Reinforcement Learning.

#### What is batch learning and online learning?

Another criterion used to classify Machine Learning systems is whether or not system can learn incrementally from a stream of incoming data. 

In __Batch Learning__, the system is incapable of learning incrementally, it must be trained using all the available data. This will generally take a lot of time and computing resources, so it is typically done offline. First the system is trained, and then it is launched into production and runs without learning anymore; it just applied what it has learned. This is called _offline learning_.

If you want a batch learning system to know about the new data, you need to train a new version of the system from scratch on the full dataset (not just the new data but also the old data), then stop the old system and replace it with the new one. 

Fortunately, the whole process of training, evaluating and launching a Machine Learning system can be automated fairly easily, so even, a batch learning system can adapt to change. Simply, update the data and train a new version of the system from scratch as often as needed. 

This solution is simple and often works fine but training the full set of data can take many hours so you would typically train a new system only ever 24 hours or even just weekly. If your system needs to adapt to rapidly changing data, then you need a more reactive solution. 

Also training on the full set of data required a lot of computing resources (CPU, memory space, disk space, disk I/0, network I/O etc.). If you have a lot of data and you automate your system to train from scratch every day, it will end up costing you a lot of money. If the amount of data is huge, it may even be impossible to use a batch learning algorithm. 

Finally, if your system needs to be able to learn autonomously, and it has limited resources, then carrying around large amounts of training data and taking up a lot of resources to train for hours everyday is a show-stopped.

Fortunately, a better option in all these cases is to use algorithms that are capable of learning incrementally.

In __online learning__, you train the system incrementally by feeding it data instances sequentially, either individually or by small groups called _mini-batches_. Each learning step is fast and cheap, so the system can learn about new data on the fly, as it arrived. 

Online learning is great for systems that receive the data as continuous flow and need to adapt to change rapidly or autonomously. It is also a good option if you have limited computing resources: once an online learning system has learned about new data instances, it does not need them anymore so you can discard them (unless you want to be able to roll back to a previous state and 'replay' the data). This can save a huge amount of space.

Online learning algorithms can also be used to train systems on huge datasets that cannot fit in one machine's main memory (this is called _out-of-core_ learning). An out-of-core learning algorithm chops the data into mini-batches, runs a training step on that data, then repeats the process until it has run on all of the data.

One important parameter of online learning systems is how fast they should adapt to changing data: this is called _learning rate_. If you set a high learning rate, then your system will rapidly adapt to new data but it will also tend to quickly forget the old data. Conversely, if you set a low learning rate, the system will have more inertia; that is, it will learn more slowly, but it will also be less sensitive to noise in the new data or to to sequences of non-representative data points.

A big challenge with online learning if that if bad data is fed to the system, the system's performance will gradually decline. In order to reduce the risk, you need to monitor the system closely and promptly switch the learning off (and possibly revert to a previously working state) if you detect a drop in performance. You may also want to monitor the input data and react to abnormal data (e.g., using an anomaly detection algorithm).

#### What is instance-based and model-based learning?

Another way to categorize Machine Learning systems is by how they generalize.  Most Machine Learning tasks are about making predictions. This means that given a number of training examples, the system needs to be able to generalize to examples it has never seen before. Having a good performance measure on the training data is good but insufficient. True goal here is to perform well on new instances. There are two main approaches to generalization: instance-based learning and model-based learning.

Instance-based learning simply compares new data points to known data points. It is possibly the most trivial form of learning, it is simply to learn by heart, then generalizes to new cases using a similarity measure. K-nearest neighbor algorithm is a well known instance-based learning algorithm. 

Model-based learning detects patterns in the training data and build a predictive model, much like scientists do. Then we use that model to make predictions on unseen data.

#### What are the main challenges of machine learning algorithms?

* Insufficient quantity of data
* Non-representative training data
* Poor quality data (polluted data full of errors, outliers and noise)
* Irrelevant features (feature engineering, feature selection, feature extraction)
* Overfitting the training data
* Underfitting the training data

#### What are the most important unsupervised learning algorithms?

In supervised learning, the training data is unlabeled. The system tries to learn without a teacher. Here are some of the most important unsupervised learning algorithms:

* Clustering
    - k-means
    - Hierarchial Cluster Analysis
    - Expectation-Maximization
* Visualization and dimensionality reduction
    - Principal Component Analysis
    - Kernel PCA
    - Locally Linear Embedding
    - t-distributed Stochastic Neighbor Embedding (t-SNE)
* Association rule learning
    - Apriori
    - Eclat

#### What is Tensorflow?

Created by the Google Brain team, TensorFlow is  a Python-friendly open source library for numerical computation and large-scale machine learning. TensorFlow bundles together a slew of machine learning and deep learning models and algorithms. It uses Python to provide a convenient front-end API for building applications within the framework, while executing those applications in high-performance C++.

#### Why Deep Learning is important?

In deep learning we want to find a mapping function from inputs to outputs. The function to be learned should be expressible as multiple levels of composition of simpler functions where different levels of functions can be viewed as different levels of abstraction. Functions at lower levels of abstraction should be found useful for capturing some simpler aspects of data distribution, so that it is possible to first learn the simpler functions and then compose them to learn more abstract concepts. It is all about learning hierarchical representations: low-level features, mid-level representations, high level concepts. Animals and humans do learn this way with simpler concepts earlier in life, and higher-level abstractions later, expressed in terms of previously learned concepts. 

#### What are the three respects of an learning algorithm to be efficient?

We want learning algorithm to be efficient in three main dimensions:

* __computational__: the amount of computing resources required to reach a given level of performance.
* __statistical__: the amount of training data required (especially labeled data) for good generalizations.
* __human involvement__: the amount of human effort (labor) required to tailor the algorithm to a task, i.e., specify the prior knowledge built into the model before training (explicitly or implicitly through engineering designs with a human-in-the-loop). 

#### What are the differences between a parameter and a hyperparameter?

A hyperparameter, also called a tuning parameter,
* is external to the model.
* cannot be estimated from data.
* is often specified by the researcher.
* is often set using heuristics.
* is often tuned for a given predictive modeling problem.
* is often used in a process to help estimate the model parameters.

An example of an hyper parameter is the learning rate for training a neural network.

A parameter 

* is a configuration variable that is internal to the model and whose value is estimated directly from data.
* is often not set manually by the researcher.
* is often saved as a part of the learned model.

An example of a parameter is the weights in a neural network.

Note that if you have to specify a model parameter manually, then it is probably a model parameter. 

#### Why do we have three sets: training, validation and test?

In practice, we have three distinct sets of labeled examples:

* training set
* validation set, and
* test set

Once you have your annotated dataset, the first thing you do is to shuffle the examples and split the data set into three subsets. Training set is usually the biggest one, you use it to build model. Validation and test sets are roughly the same sizes, much smaller than the size of the training set. The learning algorithm cannot use these examples from these two subsets to build the model. That is why those two subsets are often called _holdout sets_. 

There is no optimal proportion to split the dataset into three subsets. The reason why we have three sets and not one is because we do not want the model to do well at predicting labels of examples the learning algorithm has already seen. A trivial algorithm that simply memorizes all the training examples and then uses the memory to "predict" their labels will make no mistakes when asked to predict the labels of the training examples but such an algorithm would be useless in practice. What we really want is a model that is good at predicting examples that the learning algorithm did not see: we want good performance on a holdout set.

Why do we need two holdout sets and not one? We use the validation set to (1) to choose the learning algorithm and (2) find the best values of hyperparameters. We use then the test set to assess the model before delivering it to the client or putting it in production. 

#### What are the goals to build a learning machine?

If our goal is to build a learning machine, our research should concentrate on devising learning models with following features:

* A highly flexible way to specify prior knowledge, hence a learning algorithm that can function with a large repertoire of architectures.
* A learning algorithm that can deal with deep architectures, in which a decision involves the manipulation of many intermediate concepts, and multiple levels of non-linear steps.
* A learning algorithm that can handle large families of functions, parameterized with million of individual parameters.
* A learning algorithm that can be trained efficiently even, when the number of training examples becomes very large. This excludes learning algorithms requiring to store and iterate multiple times over the whole training set, or for which the amount of computations per example increases as more examples are seen. This strongly suggest the use of on-line learning.
* A learning algorithm that can discover concepts that can be shared easily among multiple tasks and multiple modalities (multi-task learning), and that can take advantage of large amounts of unlabeled data (semi-supervised learning).

#### What are the solutions of overfitting?

There are several solutions to the problem of overfitting:

* We can try a simpler model because in the case of overfitting, the model might be complex for the dataset, e.g., linear instead of polynomial regression, or SVM with a linear kernel instead of radial basis function, a neural network with fever layers/units.
* We can reduce the dimensionality of the dataset (removing some  irrelevant features, or using one of the dimensionality reduction techniques. Even some algorithms have built-in feature selection.)
* We can add more training data. This should reduce variance, but will have no effect on bias. More data can even make bias worse - it gives your model the chance to give highly precise, wrong answers
* We can try to use early stopping in order to prevent over-training by monitoring model performance. It is probably the most commonly used form of regularization in deep learning. Its popularity is due both to its effectiveness and its simplicity. In the case of neural networks, while the network seems to get better and better, i.e., the error on the training set decreases, at some point during training it actually begins to get worse again, i.e., the error on unseen examples increases. Early stopping may underfit by stopping too early.
* We can use regularization methods. For example, you could prune a decision tree, use dropout on a neural network, or add a penalty parameter (L1/L2 Regularization) to the cost function in regression.
* We can use Ensembling methods (Bagging and Boosting). Ensembles are machine learning methods for combining predictions from multiple separate models. Bagging uses complex base models and tries to "smooth out" their predictions, while boosting uses simple base models and tries to "boost" their aggregate complexity.
* Cross-validation is a powerful preventative measure against overfitting. Cross-validation simply repeats the experiment multiple times, using all the different parts of the training set as unseen data which we use to validate the model. This gives a more accurate indication of how well the model generalizes to unseen data.  Cross-validation does not prevent overfitting in itself, but it may help in identifying a case of overfitting.

#### Is it better to design robust or accurate algorithms?

* The ultimate goal is to design systems with good generalization capacity, that is, systems that correctly identify patterns in data instances not seen before
* The generalization performance of a learning system strongly depends on the complexity of the model assumed
* If the model is too simple, the system can only capture the actual data regularities in a rough manner. In this case, the system has poor generalization properties and is said to suffer from underfitting
* By contrast, when the model is too complex, the system can identify accidental patterns in the training data that need not be present in the test set. These spurious patterns can be the result of random fluctuations or of measurement errors during the data collection process. In this case, the generalization capacity of the learning system is also poor. The learning system is said to be affected by overfitting
* Spurious patterns, which are only present by accident in the data, tend to have complex forms. This is the idea behind the principle of Occam’s razor for avoiding overfitting: simpler models are preferred if more complex models do not significantly improve the quality of the description for the observations
* Quick response: Occam’s Razor. It depends on the learning task. Choose the right balance
* Ensemble learning can help balancing bias/variance (several weak learners together = strong learner)

#### What is feature engineering?

Feature engineering is the process of taking a dataset and constructing explanatory variables (features) that can be used to train a machine learning model for a prediction problem. Often, data is spread across multiple tables and must be gathered into a single table with rows containing the observations and features in the columns.

Traditional approach to feature engineering is to build features one at a time using domain knowledge, a tedious, time-consuming and error-prone process known as manual engineering. The code for manual feature engineering is a problem-dependent and must be written for each new dataset.

#### What are some feature scaling (a.k.a data normalization) techniques? When should you scale your data? Why?

Feature scaling is the method used to standardize the range of features of data. Since the range of values of data may vary widely, it becomes a necessary step in data processing while using ML algorithms. 

* **Min-Max Scaling**: You transform the data such that the features are within a specific range, e.g. [0,1]
     \begin{equation}
         X^{'} = \frac{X- X_{min}}{X_{max} - X_{min}}
     \end{equation}
     where $X^{'}$ is the normalized value. Min-max normalization has one fairly significant downside: it does not handle outliers very well. For example, if you have 99 values between 0 and 40, and one value is 100, then the 99 values will all be transformed to a value between 0 and 0.4. But 100 will be squished into 1, meaning that that data is just as squished as before, still an outlier!
* **Normalization**: The point of normalization is to change your observations so they can be described as a normal distribution.
     \begin{equation}
         X^{'} = \frac{X- X_{mean}}{X_{max} - X_{min}}
     \end{equation}
     All the values will be between 0 and 1. 
* **Standardization**: Standardization (also called z-score normalization) transforms your data such that the resulting distribution has a mean 0 and a standard deviation 1. 
     \begin{equation}
         X^{'} = \frac{X- X_{mean}}{\sigma}
     \end{equation}
     where $X$ is the original feature vector, $X_{mean}$ is the mean of the feature vector, and $\sigma$ is its standard deviation. Z-score normalization is a strategy of normalizing data that avoids the outlier issue of Min-Max Scaling. The only potential downside is that the features aren’t on the exact same scale.
     
 You should scale your data,
 
* when your algorithm will weight each input, e.g. gradient descent used by many neural networks, or use distance metrics, e.g., kNN, model performance can often be improved by normalizing, standardizing, otherwise scaling your data so that each feature is given relatively equal weight.
* It is also important when features are measured in different units, e.g. feature A is measured in inches, feature B is measured in feet, and feature C is measured in dollars, that they are scaled in a way that they are weighted and/or represented equally.
     In some cases, efficacy will not change but perceived feature importance might change, e.g., coefficients in a linear regression.
* Scaling your data typically does not change the performance or feature importance for tree-based models which are not distance based models, since the split points will simply shift to compensate for the scaled data. 

#### What are the types of feature selection methods?

 There are three types of feature selection methods
 
* **Filter Methods**: Feature Selection is done independent of the learning algorithm before any modeling is done. One example is finding the correlation between every feature and the target and throwing out those that do not meet a threshold. Easy, fast but naive and not as performant as other methods.
* **Wrapper Methods**: Train models on subsets of features and use the subset that results in the best performance. Examples are Stepwise or Recursive Feature selection. Advantages are that it considers each feature in the context of other features but can be computationally expensive.
* **Embedded Methods**: Learning algorithms have built-in feature selection, e.g., L1-Regularization.

#### How can you prove that one improvement you've brought to an algorithm is really an improvement over not doing anything?

You can always check the model performance after adding or removing a features, if the performance of model is dropping or improving you can see if the inclusion of that variable makes sense or not. Apart from that, you tweak different inbuilt model parameters like you increase number of trees to grow or number of iterations to do in random forest, you add a regularisation term in linear regression, you change threshold parameters in logistic regression, you assign weights to several algorithms, if you compare the accuracies and other statistics before and after making such change to model, you can understand if these result into any improvement or not.

#### What are the hyperparameter tuning methods?
Hyperparameters are not optimized by the learning algorithm itself. The researcher has to tune the hyperparameters by experimentally finding the best combination o fvalues, one per hyper parameter.

One typical way to do that, when you have enough data to have a decent validation set and the number of hyperparameters and their range is not too large is to use __grid search__.

Grid search is the most simple hyperparameter tuning technique. It builds a model for every combination of hyperparameters specified and evaluates each model. Finally, you keep the model that performs the best according to the metric. Once the best pair of hyperparameters ois found, you can try to explore the values close to the best ones in some region around them. Sometimes, this can result in an even better model. Finally you assess the selected model using the test set.

However, trying all combinations of hyperparameters, especially if there are more than a couple of them, could be time-consuming, especially for large datasets. There are more efficient techniques seuch as __random search__ and __Bayesian hyperparameter optimization__. 

Random search differs from grid search in that you no longer provide a discrete set of values to explore for each hyperparameter. Instead, you provide a statistical distribution for each hyperparameter from which values are randomly sampled and set the total number of combinations (number of iterations) you want to try.

Bayesian techniques differ from random or grid search in that they use past evaluation results to choose the next values to evaluate. The idea is to limit the number of expensive optimizations of the objective function by choosing the next hyperparameter values based on those that have done well in the past. 

There are also __gradient-based techniques__, __evolutionary optimization techniques__, and other algorithmic hyperparameter tuning techniques. 

#### How do we use probability in Machine Learning/Deep Learning framework?

1. **Class Membership Requires Predicting a Probability**: Classification predictive modeling problems are those where an example is assigned a given label. Therefore, we model the problem as directly assigning a class label to each observation (hard class classification). A more common approach is to frame the problem as a probabilistic class membership, where the probability of an observation belonging to each known class is predicted (soft class classification. Therefore,  this probability is more explicit for the network. Framing the problem as a prediction of class membership simplifies the modeling problem and makes it easier for a model to learn. It allows the model to capture ambiguity in the data, which allows a process downstream, such as the user to interpret the probabilities in the context of the domain. The probabilities can be transformed into a hard class label by choosing the class with the largest probability. The probabilities can also be scaled or transformed using a probability calibration process.

2. **Some Algorithms Are Designed Using Probability**: There are algorithms that are specifically designed to harness the tools and methods from probability. Naive Bayes, Probabilistic Graphical Models, Bayesian Belief Networks are three of those algorithms. 

3. **Models Can Be Tuned With a Probabilistic Framework**: It is common to tune the hyperparameters of a machine learning model, such as k for kNN or the learning rate in a neural network. Typical approaches include grid searching ranges of hyperparameters or randomly sampling hyperparameter combinations. Bayesian optimization is a more efficient to hyperparameter optimization that involves a directed search of the space of possible configurations based on those configurations that are most likely to result in better performance. As its name suggests, the approach was devised from and harnesses Bayes Theorem when sampling the space of possible configurations.

4. **Models Are Trained Using a Probabilistic Framework**: Many machine learning models are trained using an iterative algorithm designed under a probabilistic framework. Perhaps the most common is the framework of maximum likelihood estimation. This is the framework that underlies the ordinary least squares estimate of a linear regression model. For models that predict class membership, maximum likelihood estimation provides the framework for minimizing the difference or divergence between an observed and predicted probability distribution. This is used in classification algorithms like logistic regression as well as deep learning neural networks. t is common to measure this difference in probability distribution during training using entropy, e.g. via cross-entropy. Entropy, and differences between distributions measured via KL divergence, and cross-entropy are from the field of information theory that directly build upon probability theory. For example, entropy is calculated directly as the negative log of the probability.

5. **Probabilistic Measures Are Used to Evaluate Model Skill**: For those algorithms where a prediction of probabilities is made, evaluation measures are required to summarize the performance of the model, such as AUC-ROC curve along with confusion matrix. Choice and interpretation of these scoring methods require a foundational understanding of probability theory.

#### What are the differences and similarities between Ordinary Least Squares Estimation and Maximum Likelihood Estimation methods?

Ordinary Least Squares (OLS) tries to answer the question "What estimates minimize the squared error of the predicted values from observed?", whereas Maximum Likelihood answers the question "What estimates maximize the likelihood function?". 

The ordinary least squares, or OLS, can also be called the linear least squares. This is a method for approximately determining the unknown parameters located in a linear regression model.

Maximum likelihood estimation, or MLE, is a method used in estimating the parameters, that are most likely to produce observed data, of a statistical model and for fitting a statistical model to data.

From Wikipedia, OLS chooses the parameters of a linear function of a set of explanatory variables by the principle of least squares: minimizing the sum of the squares of the differences between the observed dependent variable (values of the variable being predicted) in the given dataset and those predicted by the linear function. Maximum Likelihood Estimation (MLE) is a method of estimating the parameters of a distribution by maximizing a likelihood function, so that under the assumed statistical model the observed data is most probable. 

The OLS estimator is identical to the maximum likelihood estimator (MLE) under the normality assumption for the error terms

Let’s recall the simple linear regression model: $y_{i} = \alpha + \beta x_{i} + \varepsilon_{i}$ where the noise variables $\varepsilon_{i}$ all have the same expectation (0) and the same variance ($\sigma^{2}$), and $Cov[\varepsilon_{i}, \varepsilon_{j}] = 0$ (unless $i = j$, of course). This is a statistical model with two variables $X$ and $Y$, where we try to predict $Y$ from $X$. We also assume that errors follow normal distribution:

$$
f(x; \mu, \sigma^{2}) = \dfrac{1}{\sqrt{2\pi \sigma^{2}}} exp\left\{-\dfrac{(x-\mu)^{2}}{2 \sigma^{2}}  \right\}
$$

We know that $E(y_{i} \mid x_{i}) = \mu_{y_{i}} = \alpha +\beta x_{i}$. The mean of the conditional distribution of $Y$ depends on the value of $X$. Indeed, that's kind of the point of a regression model. We also know that $Var(y_{i} \mid x_{i}) = \sigma_{y_{i}}^{2} = \sigma^{2}$, since $x_{i}$ is a single fixed value.

Let's write the likelihood function for this linear model:

$$
\begin{split}
L(\alpha, \beta) &= \prod_{i=1}^{n} p(y_{i} \mid x_{i};\alpha, \beta) \\
&= \prod_{i=1}^{n}  \dfrac{1}{\sqrt{2\pi \sigma_{y_{i}}^{2}}} exp\left\{-\dfrac{(y_{i}-\mu_{y_{i}})^{2}}{2 \sigma_{y_{i}}^{2}}  \right\}\\
&= \dfrac{1}{\left(2\pi \sigma^{2} \right)^{n/2}} \prod_{i=1}^{n} exp\left\{-\dfrac{(y_{i} - \alpha - \beta x_{i})^{2}}{2 \sigma^{2}}  \right\}\\
& = \dfrac{1}{\left(2\pi \sigma^{2} \right)^{n/2}} exp\left\{- \dfrac{1}{2 \sigma^{2}} \sum_{i=1}^{n} \left(y_{i} - \alpha - \beta x_{i}\right)^{2}\right\}\\
\end{split}
$$

Obviously, maximizing this likelihood is equivalently minimizing,

$$
 \sum_{i=1}^{n} \left(y_{i} - \alpha - \beta x_{i}\right)^{2}
$$

which is nothing but the sum of squares of differences between observed and predicted values. 

#### Do you suggest that treating a categorical variable as continuous variable would result in a better predictive model?

For better predictions, categorical variable can be considered as a continuous variable only when the variable is ordinal in nature.

#### Considering the long list of machine learning algorithm, given a data set, how do you decide which one to use?

Choosing a machine learning algorithm can be a difficult task. If you have much time, you can try all of them. However, usually the time you have to solve a problem is limited. You can ask yourself several questions before starting to work on the problem. Depending on your answers, you can shortlist some algorithms and try them on your data.

1. **Explainability**: Most very accurate learning algorithms are so-called "black boxes." They learn models that make very few errors, but why a model made a specific prediction could be very hard to understand and even harder to explain. Examples of such models are neural networks or ensemble models. On the other hand, kNN, linear regression, or decision tree learning algorithms produce models that are not always the most accurate, however, the way they make their prediction is very straightforward.

2. **In-memory vs. out-of-memory**: Can your dataset be fully loaded into the RAM of your server or personal computer? If
yes, then you can choose from a wide variety of algorithms. Otherwise, you would prefer incremental learning algorithms that can improve the model by adding more data gradually.

3. **Number of features and examples**: Number of training examples in the dataset and number of features to be handled by the algorithm can be troublesome for some. Some algorithms, including neural networks and gradient boosting, can handle a huge number of examples and millions of features. Others, like SVM, can be very modest in their capacity.

4. **Categorical vs. numerical features**: Depending on the data composed of categorical only, or numerical only features, or a mix of both, some algorithms cannot handle your dataset directly, and you would need to convert your categorical features into numerical ones.

5. **Nonlinearity of the data**: If the data is linearly separable or if it can be be modeled using a linear model, SVM with the linear kernel, logistic or linear regression can be good choices. Otherwise, deep neural networks or ensemble algorithms, might work better. Additionally, if you given to work on unstructured data such as images, audios, videos, then neural network would help you to build a robust model.

6. **Training speed**: Neural networks are known to be slow to train, even with GPU. Simple algorithms like logistic and linear regression or decision trees are much faster. Specialized libraries contain very efficient implementations of some algorithms; you may prefer to do research online to find such libraries. Some algorithms, such as random forests, benefit from the availability of multiple CPU cores, so their model building time can be significantly reduced on a machine with dozens of cores.

7. **Prediction speed**: The time spent for generating predictions is also considerably important for choosing the algorithm. Algorithms like SVMs, linear and logistic regression, and (some types of) neural networks, are extremely fast at the prediction time. Others, like kNN, ensemble algorithms, and very deep or recurrent neural networks, are slower. 

#### What is selection bias?

Selection bias is the bias introduced by the selection of individuals, groups or data for analysis in such a way that proper randomization is not achieved, thereby ensuring that the sample obtained is not representative of the population intended to be analyzed. It is sometimes referred to as the selection effect. The phrase "selection bias" most often refers to the distortion of a statistical analysis, resulting from the method of collecting samples. If the selection bias is not taken into account, then some conclusions of the study may be false.

The types of selection bias include:

1. **Sampling bias**: It is a systematic error due to a non-random sample of a population causing some members of the population to be less likely to be included than others resulting in a biased sample.

2. **Time interval**: A trial may be terminated early at an extreme value (often for ethical reasons), but the extreme value is likely to be reached by the variable with the largest variance, even if all variables have a similar mean.

3. **Data**: When specific subsets of data are chosen to support a conclusion or rejection of bad data on arbitrary grounds, instead of according to previously stated or generally agreed criteria.

4. **Attrition**: Attrition bias is a kind of selection bias caused by attrition (loss of participants) discounting trial subjects/tests that did not run to completion.

#### What’s the difference between a generative and discriminative model?

Disriminative models learn the explicit (hard or soft) boundaries between classes (and not necessarily in a probabilistic manner). Generative models learn the distribution of individual classes, therefore, providing a model of how the data is actually generated, in terms of a probabilistic model. (e.g., logistic regression, support vector machines or the perceptron algorithm simply give you a separating decision boundary, but no model of generating synthetic data points). For more details, you can read [this blog post](https://mmuratarat.github.io/2019-08-23/generative-discriminative-models){:target="_blank"}.

#### What cross-validation technique would you use on a time series dataset?

Instead of using standard k-folds cross-validation, you have to pay attention to the fact that a time series is not randomly distributed data — it is inherently ordered by chronological order. When the data are not independent, cross-validation becomes more difficult as leaving out an observation does not remove all the associated information due to the correlations with other observations.

The "canonical" way to do time-series cross-validation is cross-validation on a rolling basis, i.e., "roll" through the dataset. Start with a small subset of data for training purpose, forecast for the later data points and then check the accuracy for the forecasted data points. The same forecasted data points are then included as part of the next training dataset and subsequent data points are forecasted.

To make things intuitive, here is an image for 5-fold CV:

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/cv_for_time_series.png?raw=true)

The forecast accuracy is computed by averaging over the test sets. This procedure is sometimes known as "evaluation on a rolling forecasting origin" because the "origin" at which the forecast is based rolls forward in time. For more details:

1. [https://robjhyndman.com/hyndsight/tscv/](https://robjhyndman.com/hyndsight/tscv/){:target="_blank"}
2. [https://robjhyndman.com/hyndsight/crossvalidation/](https://robjhyndman.com/hyndsight/crossvalidation/){:target="_blank"}

#### What is the difference between "long" and "wide" format data?

In the wide format, a subject’s repeated responses will be in a single row, and each response is in a separate column. In the long format, each row is a one-time point per subject. You can recognize data in wide format by the fact that columns generally represent groups.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/wide_long_format_data.png?raw=true)

#### Can you cite some examples where a false positive is important than a false negative, and where a false negative important than a false positive, and where both false positive and false negatives are equally important?

False Positives are the cases where you wrongly classified a non-event as an event (a.k.a Type I error). False Negatives are the cases where you wrongly classify events as non-events (a.k.a Type II error).

An example of where a false positive is important than a false negative is that in the medical field, assume you have to give chemotherapy to patients. Assume a patient comes to that hospital and he is tested positive for cancer, based on the lab prediction but he actually doesn’t have cancer. This is a case of false positive. Here it is of utmost danger to start chemotherapy on this patient when he actually does not have cancer. In the absence of cancerous cell, chemotherapy will do certain damage to his normal healthy cells and might lead to severe diseases, even cancer

An example of where a false negative important than a false positive is that what if Jury or judge decides to make a criminal go free?

An example of where both false positive and false negatives are equally important is that, in the Banking industry giving loans is the primary source of making money but at the same time if your repayment rate is not good you will not make any profit, rather you will risk huge losses. Banks don't want to lose good customers and at the same point in time, they don’t want to acquire bad customers. In this scenario, both the false positives and false negatives become very important to measure

#### Describe the difference between univariate, bivariate and multivariate analysis?

Univariate analysis is the simplest form of data analysis where the data being analyzed contains only one variable. Since it's a single variable it doesn’t deal with causes or relationships.  The main purpose of univariate analysis is to describe the data and find patterns that exist within it

You can think of the variable as a category that your data falls into. One example of a variable in univariate analysis might be "age". Another might be "height". Univariate analysis would not look at these two variables at the same time, nor would it look at the relationship between them.  

Some ways you can describe patterns found in univariate data include looking at mean, mode, median, range, variance, maximum, minimum, quartiles, and standard deviation. Additionally, some ways you may display univariate data include frequency distribution tables, bar charts, histograms, frequency polygons, and pie charts.

Bivariate analysis is used to find out if there is a relationship between two different variables. Something as simple as creating a scatterplot by plotting one variable against another on a Cartesian plane (think X and Y axis) can sometimes give you a picture of what the data is trying to tell you. If the data seems to fit a line or curve then there is a relationship or correlation between the two variables.  For example, one might choose to plot caloric intake versus weight.

Multivariate analysis is the analysis of three or more variables.  There are many ways to perform multivariate analysis depending on your goals.  Some of these methods include Additive Tree, Canonical Correlation Analysis, Cluster Analysis, Correspondence Analysis / Multiple Correspondence Analysis, Factor Analysis, Generalized Procrustean Analysis, MANOVA, Multidimensional Scaling, Multiple Regression Analysis, Partial Least Square Regression, Principal Component Analysis / Regression / PARAFAC,  and Redundancy Analysis.

## Deep Learning

#### What is an epoch, a batch and an iteration?

A batch is the complete dataset. Its size is the total number of training examples in the available dataset.

Mini-batch size is the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.

A Mini-batch is a small part of the dataset of given mini-batch size.
One epoch = one forward pass and one backward pass of all the training examples

Number of iterations = An iteration describes the number of times a batch of data passed through the algorithm, number of passes, each pass using [batch size] number of examples. To be clear, one pass = one forward pass + one backward pass (we do not count the forward pass and backward pass as two different passes).

_Example_: if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch.

#### What is deep learning?

Deep Learning is an area of Machine Learning that attempts to mimic the activity in the layers of neurons in the brain to learn how to recognize the complex patterns in the data. The "deep" in deep learning refers to the large number of layers of neurons in contemporary ML models that help to learn rich representations of data to achieve better performance gains.

#### Why are deep networks better than shallow ones?

Both the Networks, be it shallow or deep are capable of approximating any function. However, a shallow network works with only a few features, as it can’t extract more. 

Deep learning's main distinguishing feature from "Shallow Learning" is that deep learning methods derive their own features directly from data (feature engineering), while shallow learning relies on the handcrafted features based upon heuristics of the target problem. Deep learning architecture can learn representations and features directly from the input with little to no prior knowledge. 

Deep networks have several hidden layers often of different types so they are able to build or create or extract better features/concepts than shallow models with fewer parameters. It makes your network more eager to recognize certain aspects of input data. 

Deep learning can learn multiple levels of representations that correspond to different levels of abstraction; the levels form  a hierarchy of concepts. It is all about learning hierarchical representations: low-level features, mid-level representations, high level concepts. Higher level concepts are derived from lower level features. Animals and humans do learn this way with simpler concepts earlier in life, and higher-level abstractions later, expressed in terms of previously learned concepts.

#### What is a perceptron?

#### What are the shortcomings of a single layer perceptron?

There are two major problems:

* Single-layer Perceptrons cannot classify non-linearly separable data points.
* Complex problems, that involve a lot of parameters cannot be solved by Single-Layer Perceptrons

#### What does a neuron compute?
An artificial neuron calculates a "weighted sum" of its input, adds a bias ($z = Wx+b$), followed by an activation function.

#### What is the role of activation functions in a Neural Network?

The goal of an activation function is to introduce nonlinearity into the neural network so that it can learn more complex function i.e. converts the processed input into an output called the activation value. Without it, the neural network would be only able to learn function which is a linear combination of its input data.

If we do not apply an activation function then the output signal would simply be a simple linear function. A linear function is just a polynomial of one degree. 

$$
z=\beta_{0}1 + \beta_{1}x_{1} + \beta_{2}x_{2} + \ldots +\beta_{p}x_{p}\,\,\,\,\, \mathbf{(a)}
$$

Each input variable $x_{j}$ is represented with a node and each parameter $\beta_{j}$ with a link. Furthermore, the output $z$ is described as the sum of all terms $\beta_{j}x_{j}$. Note that we use 1 as the input variable corresponding to the bias term (a.k.a. offset term) $\beta_{0}$. 

To describe _nonlinear_ relationship between $x = \left[1\,\,x_{1}\,\,x_{2}\,\, \ldots \,\,x_{p}\right]^{T}$ and $z$, we introduce a nonlinear scalar-valued function called _activation function_ $\sigma: \mathbb{R} \to \mathbb{R}$. The linear regression model is now modified into a _generalized_ linear regression model where the linear combination of the inputs is squashed through the (scalar) activation function. 

$$
z = \sigma \left( \beta_{0}1 + \beta_{1}x_{1} + \beta_{2}x_{2} + \ldots +\beta_{p}x_{p} \right)\,\,\,\,\, \mathbf{(b)}
$$

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/glm_activation.png?raw=true)

Now, a linear equation is easy to solve but they are limited in their complexity and have less power to learn complex functional mappings from data. A neural network without activation function would simply be a linear regression model, which has limited power and does not performs good most of the times. Therefore, we make further extensions to increase the generality of the model by sequentially stacking these layers. 

Another important feature of an activation function is also that it should be differentiable. We need it to be this way so as to perform backpropogation optimization strategy while propogating backwards in the network to compute gradients of error (loss) with respect to parameters (weights/biases) and then accordingly optimize weights using Gradient Descent algorithm or any other optimization technique to reduce error.

####  How many types of activation functions are there ?


#### What is a Multi-Layer-Perceptron

#### What is Deep Neural Network?

Deep neural networks can model complicated relationships and is one of the state of art methods in machine learning as of today.

We enumerate layers with index $l$. Each _layer_ is parameterized with a weight matrix $\mathbf{W}^{(l)}$ and a bias vector $\mathbf{b}^{(l)}$. For example, $\mathbf{W}^{(1)}$ and $\mathbf{b}^{(1)}$ belong to layer $l=1$; $\mathbf{W}^{(2)}$ and $\mathbf{b}^{(2)}$ belong to layer $l=2$ and so forth. We also have multiple layers of hidden units denoted by $\mathbf{h}^{(l-1)}$. Each such layer consists of $M_{l}$ hidden units, 

$$
\mathbf{h}^{(l)} = \left[\mathbf{h}^{(l)}_{1}, \ldots, \mathbf{h}^{(l)}_{M_{l}} \right]^{T}
$$

where the dimensions $M_{1}, M_{2}, \ldots$ can be different for different layers.

Each layer maps a hidden layer $\mathbf{h}^{(l-1)}$ to the next hidden layer $\mathbf{h}^{(l)}$ as:

$$
\mathbf{h}^{(l)} = \sigma \left(\mathbf{W}^{(l)T}\mathbf{h}^{(l-1)} + \mathbf{b}^{(l)T} \right)
$$

This means that the layers are stacked such that the output of the first layer $\mathbf{h}^{(1)}$ (the first layer of hidden units) is the input to the second layer, the output of the second layer $\mathbf{h}^{(2)}$ (the second layer of hidden units) is the input to the third layer, etc. By stacking multiple layers we have constructed a deep neural network. A deep neural network of $L$ layers can mathematically be described as:

$$
\begin{split}
\mathbf{h}^{(1)} &= \sigma \left(\mathbf{W}^{(1)T}\mathbf{x} + \mathbf{b}^{(1)T} \right)\\
\mathbf{h}^{(2)} &= \sigma \left(\mathbf{W}^{(2)T}\mathbf{h}^{(1)} + \mathbf{b}^{(2)T} \right)\\
&\cdots \\
\mathbf{h}^{(L-1)} &= \sigma \left(\mathbf{W}^{(L-1)T}\mathbf{h}^{(L-2)} + \mathbf{b}^{(L-1)T} \right)\\
z &= \mathbf{W}^{(L)T}\mathbf{h}^{(L-1)} + \mathbf{b}^{(L)T}\\
\end{split}
$$

The weight matrix $\mathbf{W}^{(1)}$ for the first layer $l = 1$ has the dimension $p \times M_{1}$ and the corresponding bias vector $\mathbf{b}^{(1)}$ the dimension $1 \times M_{1}$. In deep learning it is common to consider applications where also the output is multi-dimensional $\mathbf{z} = \left[z_{1}, \ldots, z_{K}\right]^{T}$. This means that for the last layer the weight matrix $\mathbf{W}^{(L)}$ has the dimension $M_{L−1} \times K$ and the bias vector $\mathbf{b}^{(L)}$ the dimension $1 \times K$. For all intermediate layers $l = 2, \ldots, L − 1$, $\mathbf{W}^{(l)}$ has the dimension $M_{l−1} \times M_{l}$ and the corresponding bias vector $1 \times M_{l}$.

The number of inputs $p$ and the number of outputs $K$ (number of classes) are given by the problem, but the number of layers
$L$ and the dimensions $M_{1}, M_{2},\ldots$ are user design choices that will determine the flexibility of the model.


#### What is softmax function and when we use it?

The softmax function is used in various multiclass classification methods. It takes an un-normalized vector, and normalizes it into a probability distribution. It is often used in neural networks, to map the non-normalized output to a probability distribution over predicted output classes. It is a function which gets applied to a vector in $z \in R^{K}$ and returns a vector in $[0,1] ^{K}$ with the property that the sum of all elements is 1, in other words, the softmax function is useful for converting an arbitrary vector of real numbers into a discrete probability distribution:

$$
softmax(z_j) = \frac{e^{z_{j}}}{\sum_{j=1}^K e^{z_{j}}} \;\;\;\text{ for } j=1, \dots, K
$$

where $\mathbf{z} = \left[z_{1}, \ldots, z_{K}\right]^{T}$. The inputs to the softmax function, i.e., the variables $z_{1}, z_{2}, \ldots, z_{K}$ are referred to as _logits_.

Intiutively, the softmax function is a "soft" version of the maximum function. A "hardmax" function (i.e. argmax) is not differentiable. The softmax gives at least a minimal amount of probability to all elements in the output vector, and so is nicely differentiable. Instead of selecting one maximal element in the vector, the softmax function breaks the vector up into parts of a whole (1.0) with the maximal input element getting a proportionally larger chunk, but the other elements get some of it as well. Another nice property of it, the output of the softmax function can be interpreted as a probability distribution, which is very useful in Machine Learning because all the output values are in the range of (0,1) and sum up to $1.0$. This is especially useful in multi-class classification because we often want to assign probabilities that our instance belong to one of a set of output classes.

For example, let's consider we have 4 classes, i.e. $K=4$, and unscaled scores (logits) are given by $[2,4,2,1]$. The simple argmax function outputs $[0,1,0,0]$. The argmax is the goal, but it's not differentiable and we can't train our model with it. A simple normalization, which is differentiable, outputs the following probabilities $[0.2222,0.4444,0.2222,0.1111]$. That's really far from the argmax! Whereas the softmax outputs $[0.1025,0.7573,0.1025,0.0377]$. That's much closer to the argmax! Because we use the natural exponential, we hugely increase the probability of the biggest score and decrease the probability of the lower scores when compared with standard normalization. Hence the "max" in softmax.

Softmax is fundamentally a vector function. It takes a vector as input and produces a vector as output. In other words, it has multiple inputs and outputs.

####  What is the cost function? 

In predictive modeling, cost functions are used to estimate how badly models are performing. Put it simply, a cost function is a measure of how wrong the model is in terms of its ability to estimate the relationship between X and y. This is typically expressed as a difference or distance between the predicted value and the actual value. The cost function (you may also see this referred to as loss or error) can be estimated by iteratively running the model to compare estimated predictions against "ground truth", i.e., the known values of $y$.

The objective here, therefore, is to find parameters, weights/biases or a structure that minimizes the cost function.

The terms cost function and loss function are synonymous, some people also call it error function.

However, there are also some different definitions out there. The loss function computes the error for a single training example, while the cost function will be average over all data points.

#### What is cross-entropy? How we define the cross-entropy cost function?

Entropy is a measure of the uncertainty associated with a given distribution $p(y)$. When we have $K$ classes, we compute the entropy of a distribution, using the formula below

$$
H(p) = - \sum_{k=1}^{K} p(y_{k}) \log p(y_{k})
$$

(This can also be thought as in the following. There are K distinct events. Each event $K$ has probability $p(y_{k})$)

If we know the true distribution of a random variable, we can compute its entropy. However, we cannot always know the true distribution. That is what Machine Learning algorithms do. We try to approximate the true distribution with an other distribution, say, $q(y)$.

If we compute entropy (uncertainty) between these two (discrete) distributions, we are actually computing the cross-entropy between them:

$$
H(p, q) = -\sum_{k=1}^{K} p(y_{k}) \log q(y_{k})
$$

If we can find a distribution $q(y)$ as close as possible to $p(y)$, values for both cross-entropy and entropy will match as well. However, this is not the always case. Therefore, cross-entropy will be greater than the entropy computed on the true distribution.

$$
H(p,q)−H(p) > 0
$$

This difference between cross-entropy and entropy is called _Kullback-Leibler Divergence_.

The Kullback-Leibler Divergence,or KL Divergence for short, is a measure of dissimilarity between two distributions.

$$
\begin{split}
D_{KL} (p || q) = H(p, q) - H(p) &= \mathbb{E}_{p(y_{k})} \left [ \log \left ( \frac{p(y_{k})}{q(y_{k})} \right ) \right ] \\
&= \sum_{k=1}^{K} p(y_{k}) \log\left[\frac{p(y_{k})}{q(y_{k})}\right] \\
&=\sum_{k=1}^{K} p(y_{k}) \left[\log p(y_{k}) - \log q(y_{k})\right]
\end{split}
$$

This means that, the closer $q(y)$ gets to $p(y)$, the lower the divergence and consequently, the cross-entropy will be. In other words, KL divergence gives us "distance" between 2 distributions, and that minimizing it is equivalent to minimizing cross-entropy. Minimizing cross-entropy will make $q(y)$ converge to $p(y)$, and $H(p,q)$ itself will converge to $H(p)$. Therefore, we need to approximate to a good distribution by using the classifier.

Now, for one particular data point, if $p \in \\{y, 1−y\\}$ and $q \in \\{\hat{y} ,1−\hat{y}\\}$, we can re-write cross-entropy as:

$$
H(p, q) = -\sum_{k=1}^{K=2} p(y_{k}) \log q(y_{k}) =-y\log \hat{y}-(1-y)\log (1-\hat{y})
$$

which is nothing but logistic loss.

The final step is to compute the average of all points in both classes, positive and negative, will give binary cross-entropy formula.

$$
L(\theta) = - \frac{1}{n} \sum_{i=1}^{n}  \left[y_{i} \log (p_i) + (1-y_{i}) \log (1- p_{i}) \right]
$$

where $i$ indexes samples/observations, n is the number of observations, where $y$ is the label ($1$ for positive class and $0$ for negative class) and $p(y)$ is the predicted probability of the point being positive for all $n$ points. In the simplest case, each $y$ and $p$ is a number, corresponding to a probability of one class.

Multi-class cross entropy formula is as follows:

$$
L(\theta) = - \frac{1}{n} \sum_{i=1}^{n}  \sum_{j=1}^{K} \left[y_{ij} \log (p_{ij}) \right]
$$

where $i$ indexes samples/observations and $j$ indexes classes. Here, $y_{ij}$ and $p_{ij}$ are expected to be probability distributions over $K$ classes. In a neural network, $y_{ij}$ is one-hot encoded labels and $p_{ij}$ is scaled (softmax) logits.

When $K=2$, one will get binary cross entropy formula.

#### Why don’t we use KL-Divergence in machine learning models instead of the cross entropy?

The KL-Divergence between distributions requires us to know both the true distribution and distribution of our predictions thereof. Unfortunately, we never have the former: that’s why we build a predictive model using a Machine Learning algorithm.

#### Can KL divergence be used as a distance measure?

It may be tempting to think of KL Divergence as a distance metric, however we cannot use KL Divergence to measure the distance between two distributions. The reason for this is that KL Divergence is not symmetric, meaning that $D_{KL}(p\mid \mid q)$ may not be equal to $D_{KL}(q\mid \mid p)$.

#### What is gradient descent?

Gradient descent is an optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient because the gradient points in the direction of the greatest increase of the function, that is, the direction of steepest ascent. In machine learning, we use gradient descent to update the parameters of our model. Parameters refer to coefficients in Linear Regression and weights in neural networks.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/gradient_cost.gif?raw=true)

#### Explain the following three variants of gradient descent: batch, stochastic and mini-batch?

#### What will happen if the learning rate is set too low or too high?

The size of these steps is called the learning rate. With a high learning rate we can cover more ground each step, but we risk overshooting the lowest point since the slope of the hill is constantly changing. With a very low learning rate, we can confidently move in the direction of the negative gradient since we are recalculating it so frequently. A low learning rate is more precise, but calculating the gradient is time-consuming, so it will take us a very long time to get to the bottom.

#### What is backpropagation?

#### What is Early Stopping?
It is a regularization technique that stops the training process as soon as the validation loss reaches a plateau or starts to increase.

#### Why is Weight Initialization important in Neural Networks?

#### What are the Hyperparameteres? Name a few used in any Neural Network.

Hyperparameters are the variables which determine the network structure, e.g., number of hidden units and the variables which determine how the network is trained, e.g., learning rate. Hyperparameters are set before training.

**Network Hyperparameters**:

* Number of Hidden Layers
* Network Weight Initialization
* Activation Function

**Training Hyperparameters**:

* Learning Rate
* Momentum
* Number of Epochs
* Batch Size

#### What is model capacity?

It is the ability to approximate any given function. The higher model capacity is the larger amount of information that can be stored in the network.

#### What is softmax function? What is the difference between softmax function and sigmoid function? In which layer softmax action function will be used ?
 
Softmax function and Sigmoid function can be used in the different layers of neural networks.

The softmax function is simply a generalization of the logistic function (sigmoid function) that allows us to compute meaningful class-probabilities in multi-class settings (a.k.a. MaxEnt, multinomial logistic regression, softmax Regression, Maximum Entropy Classifier). Sigmoid function is used for binary classification.
 
 In a neural network, it is mostly used in the output layer in order to have probabilistic output of the network. When performing classification you often want not only to predict the class label, but also obtain a probability of the respective label. This probability gives you some kind of confidence on the prediction. .

In the two-class logistic regression, the predicted probablies are as follows, using the sigmoid function:

$$
\begin{split}
P(y^{(i)}=1 \mid \mathbf{x}^{(i)}, \theta) &= \dfrac{1}{1+exp(-\theta^{T} \cdot \mathbf{x}^{(i)})}\\
P(y^{(i)}=0 \mid \mathbf{x}^{(i)}, \theta) &= 1 - P(y^{(i)}=1 \mid \mathbf{x}^{(i)}, \theta) = \dfrac{exp(-\theta^{T} \cdot \mathbf{x}^{(i)})}{1+exp(-\theta^{T} \cdot \mathbf{x}^{(i)})}
\end{split}
$$

In the multiclass logistic regression, with $K$ classes, the predicted probabilities are as follows, using the softmax function:

$$
P(Y_{i}=k) = \dfrac{exp(\theta_{k}^{T} \cdot \mathbf{x}^{(i)})}{\sum_{0 \leq c \leq K} exp(\theta_{c}^{T} \cdot \mathbf{x}^{(i)})}
$$

#### What’s the difference between a feed-forward and a backpropagation neural network?

A Feed-Forward Neural Network is a type of Neural Network architecture where the connections are "fed forward", i.e. do not form cycles (there is no  feedback connections like Recurrent Neural Network).  The term "Feed-Forward" is also used when information ﬂows through from input layer to output layer. It travels from input to hidden layer and from hidden layer to the output layer.

Backpropagation is a training algorithm consisting of 2 steps:

* Feed-Forward the values.
* Calculate the error and propagate it back to the earlier layers.

So to be precise, forward-propagation is part of the backpropagation algorithm but comes before backpropagating.

#### What is Dropout and Batch Normalization?

#### What is the relationship between the dropout rate and regularization?

Higher dropout rate says that more neurons are active. So there would be less regularization.

#### What is Variational dropout?

#### Name a few deep learning frameworks

* TensorFlow
* Caffe
* The Microsoft Cognitive Toolkit/CNTK
* Torch/PyTorch
* MXNet
* Chainer
* Keras

#### Explain a Computational Graph.

Everything in a tensorflow is based on creating a computational graph. It has a network of nodes where each node performs an operation, Nodes represent mathematical operations and edges represent tensors. Since data flows in the form of a graph, it is also called a “DataFlow Graph.”

#### What is a CNN?

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/putting_all_together.png?raw=true)

#### What is Pooling in CNN and how does it work? Why does  it work?

#### Explain the different Layers of CNN.

There are 4 different layers in a convolutional neural network:
1. **Convolution Layer**: The Conv layer is the core building block of a Convolutional Network that does most of the computational heavy lifting. It is the first layer to extract features from an input image.
2. **Activation Layer**: After each convolutional layer, it is convention to apply a nonlinear layer (or activation layer) immediately afterward. The purpose of this layer is to introduce nonlinearity, without affecting the receptive fields of the conv layer, to a system that basically has just been computing linear operations during the convolutional layers (just element-wise multiplications and summations). This stage is also called detector stage.
3. **Pooling Layer**: Spatial Pooling (also called subsampling or downsampling, shrink) reduces the dimensionality of each feature map but retains the most important information.  Spatial Pooling can be of different types: Max, Average, Sum etc. It is common to periodically insert a Pooling layer in-between successive layers in a architecture. Pooling is applied separately on each feature maps. Pooling neuron has no weights. All it does is to aggregate inputs using an aggregation fixed function, such as the max and mean.
4. **Fully-connected Layer (Dense Layer)**: The CNN process begins with convolution and pooling, breaking down the image into features, and analyzing them independently. The result of this process feeds into a fully connected neural network structure that drives the final classification decision. This layer is mostly used with sigmoid function (for two classes) or softmax function (for multiple classes) in order to provide the final probabilities for each label.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/components_CNN.png?raw=true)

#### How to compute the spatial size of output image after a Convolutional layer?

We don’t have to manually calculate the dimension (the spatial size) of the output, but it’s a good idea to do so to keep a mental account of how our inputs are being transformed at each step. We can compute the spatial size on each dimension (width/height/depth).

* The input volume size ($W_{1}$ and $H_{1}$, generally they are equal and $D_{1}$)
* Number of filters ($K)$
* the receptive field size of filter ($F$)
* the stride with which they are applied ($S$)
* the amount of zero padding used ($P$) on the border. 

produces a volume of size $W_{2} \times H_{2} \times D_{2}$ where:

$W_{2} = \dfrac{W_{1} - F + 2P}{S} + 1$

$H_{2} = \dfrac{H_{1} - F + 2P}{S} + 1$ (i.e., width and height are computed equally by symmetry)

$D_{2}= K$ 

* 2P comes from the fact that there should be a padding on each side.

#### What is the number of parameters in one CNN layer?

$F$ is the receptive field size of filter (kernel) and $K$ is the number of filters. $D_{1}$ is the depth (the number of channels) of the image.

In a Conv Layer, the depth of every kernel (filter) is always equal to the number of channels in the input image. So every kernel has $F^{2} \times D_{1}$ parameters, and there are $K$ such kernels.

Parameter sharing scheme is used in Convolutional Layers to control the number of parameters.

With parameter sharing, which means no matter the size of your input image, the number of parameters will remain fixed. $F \cdot F \cdot D_{1}$ weights per feature map are introduced and for a total of $(F \cdot F \cdot D_{1}) \cdot K$ weights and $K$ biases. Number of parameters of the Conv Layer is $(F \cdot F \cdot D_{1}) \cdot K + K$

#### What is an RNN?

#### What is the number of parameters in an RNN?

#### What are some issues faced while training an RNN?

#### What is Vanishing Gradient Problem?

#### What is Exploding Gradient Problem?


Exploding gradients can be dealt with by gradient clipping (truncating the gradient if it exceeds some magnitude)

ReLU in conjunction with batch normalization (or ELU or SELU) has effectively obviated both vanishing/ exploding gradients and the internal covariate shift problem.

The problem still remains for recurrent nets though (to some extent at least).

#### What are the different types of RNN?

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/0_1PKOwfxLIg_64TAO.jpeg?raw=true)

Each rectangle is a vector and arrows represent functions (e.g. matrix multiply). Input vectors are in red, output vectors are in blue and green vectors hold the RNN's state (more on this soon). From left to right: (1) Vanilla mode of processing without RNN, from fixed-sized input to fixed-sized output (e.g. image classification). (2) Sequence output (e.g. image captioning takes an image and outputs a sentence of words). (3) Sequence input (e.g. sentiment analysis where a given sentence is classified as expressing positive or negative sentiment). (4) Sequence input and sequence output (e.g. Machine Translation: an RNN reads a sentence in English and then outputs a sentence in French). (5) Synced sequence input and output (e.g. video classification where we wish to label each frame of the video). Notice that in every case are no pre-specified constraints on the lengths sequences because the recurrent transformation (green) is fixed and can be applied as many times as we like.

#### What is an LSTM cell? How does an LSTM network work? Explain the gates.

Theoretically recurrent neural network can work. But in practice, it suffers from two problems: vanishing gradient and exploding gradient, which make it unusable. 

Recurrent Neural Networks suffer from short-term memory. If a sequence is long enough, they will have a hard time carrying information from earlier time steps to later ones due to the vanishing gradient. So in recurrent neural networks, layers that get a small gradient update stops learning. Those are usually the earlier layers. Think about a recurrent neural network unrolled through time. So because these layers do not learn, RNNs can forget what it seen in longer sequences, thus having a short-term memory.

Then later, LSTM (long short term memory) was invented as the solution to short-term memory. In order to solve this issue, a memory unit, called the cell has been explicitly introduced into the network. They have internal mechanisms called gates that can regulate the flow of information.

Note that LSTM does not always protect you from exploding gradients! Therefore, successful LSTM applications typically use gradient clipping.

LSTMs are recurrent network where you replace each neuron by a memory unit. This unit contains an actual neuron with a recurrent self-connection. The activations of those neurons within memory units are the state of the LSTM network. This is the diagram of a LSTM building block

![](https://raw.githubusercontent.com/mmuratarat/mmuratarat.github.io/master/_posts/images/lstm.png)

The network takes three inputs. $X_t$ is the input of the current time step. $h_{t-1}$ is the output from the previous LSTM unit and $C_{t-1}$ is the "memory" of the previous unit. As for outputs, $h_{t}$ is the output of the current network. $C_{t}$ is the memory of the current unit.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/1_S0Y1A3KXYO7_eSug_KsK-Q.png?raw=true)
 
Equations below summarizes how to compute the cell’s long-term state, its short-term state, and its output at each time step for a single instance (the equations for a whole mini-batch are very similar).

1. Input gate:
$$ i_{t} = \sigma (W_{xi}^{T} \cdot X_{t} +  W_{hi}^{T} \cdot h_{t-1}  + b_{i})$$

2. Forget gate:
$$ f_{t} = \sigma (W_{xf}^{T} \cdot X_{t} + W_{hf}^{T} \cdot h_{t-1} + b_{f})$$

3. New Candidate:
$$ \widetilde{C}_{t} = tanh (W_{xc} \cdot X_{t} + W_{hc} \cdot h_{t-1} + b_{c})$$

4. Cell State:
$$ C_{t} = f_{t}\circ C_{t-1} + i_{t}  \circ \widetilde{C}_{t}$$

5. Output gate:
$$ o_{t} = \sigma (W_{xo} \cdot X_{t} + W_{ho} \cdot h_{t-1} + b_{o})$$

6. Hidden State:
$$ h_{t} = o_{t}\circ tanh(C_{t})$$

*  $W_{xi}$, $W_{xf}$, $W_{xc}$, $W_{xo}$ are the weight matrices of each of the three gates and block input for their connection to the input vector $X_{t}$.
*  $W_{hi}$, $W_{hf}$, $W_{hc}$, $W_{ho}$ are the weight matrices of each of the three gates and block input  for their connection to the previous short-term state $h_{t-1}$.
*  $b_{i}$, $b_{f}$, $b_{c}$ and $b_{o}$ are the bias terms for each of the three gates and block input . 
*  $\sigma$ is an element-wise sigmoid activation function of the neurons, and $tanh$ is an element-wise hyperbolic tangent activation function of the neurons
*  $\circ$ represents the Hadamard product (elementwise product).

**NOTE**: Sometimes, $h_t$ is called as the outgoing state and $c_t$ is called as the internal cell state.

Just like for feedforward neural networks, we can compute all these in one shot for a whole mini-batch by placing all the inputs at time step $t$ in an input matrix $X_{t}$. If we write down the equations for **all instances in a mini-batch**, we will have:

1. Input gate:
$$ i_{t} = \sigma (X_{t}\cdot W_{xi} + h_{t-1} \cdot W_{hi} + b_{i})$$

2. Forget gate:
$$ f_{t} = \sigma (X_{t} \cdot W_{xf} + h_{t-1} \cdot W_{hf} + b_{f})$$

3. New Candidate:
$$ \widetilde{C}_{t} = tanh (X_{t} \cdot W_{xc} + h_{t-1} \cdot W_{hc} + b_{c})$$

4. Cell State:
$$ C_{t} = f_{t}\circ C_{t-1} + i_{t}  \circ \widetilde{C}_{t}$$

5. Output gate:
$$ o_{t} = \sigma (X_{t} \cdot W_{xo} + h_{t-1} \cdot W_{ho} + b_{o})$$

6. Hidden State:
$$ h_{t} = o_{t}\circ tanh(C_{t})$$

We can concatenate the weight matrices for $X_{t}$ and $h_{t-1}$ horizontally, we can rewrite the equations above as the following:

1. Input gate:
$$ i_{t} = \sigma ( [X_{t} h_{t-1}] \cdot W_{i}  + b_{i})$$

2. Forget gate:
$$ f_{t} = \sigma ([X_{t} h_{t-1}] \cdot W_{f} + b_{f})$$

3. New Candidate:
$$ \widetilde{C}_{t} = tanh ( [X_{t} h_{t-1}] \cdot W_{c} + b_{c})$$

4. Cell State:
$$ C_{t} = f_{t}\circ C_{t-1} + i_{t}  \circ \widetilde{C}_{t}$$

5. Output gate:
$$ o_{t} = \sigma ([X_{t} h_{t-1}] \cdot W_{o}+ b_{o})$$

6. Hidden State:
$$ h_{t} = o_{t}\circ tanh(C_{t})$$

Let's denote $B$ as batch size, $F$ as number of features and $U$ as number of units in an LSTM cell, therefore, the dimensions will be computed as follows:

$X_{t} \in \mathbb{R}^{B \times F}$

$h_{t-1} \in \mathbb{R}^{B \times U}$

$h_{t} \in \mathbb{R}^{B \times U}$

$C_{t-1} \in \mathbb{R}^{B \times U}$

$W_{xi} \in \mathbb{R}^{F \times U}$

$W_{xf} \in \mathbb{R}^{F \times U}$

$W_{xc} \in \mathbb{R}^{F \times U}$

$W_{xo} \in \mathbb{R}^{F \times U}$

$W_{hi} \in \mathbb{R}^{U \times U}$

$W_{hf} \in \mathbb{R}^{U \times U}$

$W_{hc} \in \mathbb{R}^{U \times U}$

$W_{ho} \in \mathbb{R}^{U \times U}$

$W_{i} \in \mathbb{R}^{F+U \times U}$

$W_{c} \in \mathbb{R}^{F+U \times U}$

$W_{f} \in \mathbb{R}^{F+U \times U}$ 

$W_{o} \in \mathbb{R}^{F+U \times U}$ 

$b_{i} \in \mathbb{R}^{U}$

$b_{c} \in \mathbb{R}^{U}$

$b_{f} \in \mathbb{R}^{U}$

$b_{o} \in \mathbb{R}^{U}$

$i_{t} \in \mathbb{R}^{B \times U}$

$f_{t} \in \mathbb{R}^{B \times U}$

$C_{t} \in \mathbb{R}^{B \times U}$

$h_{t} \in \mathbb{R}^{B \times U}$

$o_{t} \in \mathbb{R}^{B \times U}$

**NOTE**: Batch size can be $1$. In that case, $B=1$.

1. **New temporary memory**: Use $X_{t}$ and $h_{t-1}$ to generate new memory that includes aspects of $X_{t}$.
2. **Input gate**: Use $X_{t}$ and $h_{t-1}$ to determine whether the temporary memory $\widetilde{C_{t}}$ is worth preserving.
3. **Forget gate**: Assess whether the past memory cell $C_{t-1}$ should be included in $C_{t}$.
4. **Update memory state**: Use forget and input gates to combine new temporary memory and the current memory cell state to get $C_{t}$.
5. **Output gate**: Decides which part of $C_{t}$ should be exposed to $h_{t}$. 

#### Why sigmoid function in activations of the 3 gates?

Gates contains sigmoid activations. A sigmoid activation is similar to the tanh activation. Instead of squishing values between $-1$ and $1$, it squishes values between $0$ and $1$. That is helpful to update or forget data because any number getting multiplied by $0$ is $0$, causing values to disappears or be "forgotten". Any number multiplied by $1$ is the same value therefore that value stay’s the same or is "kept". The network can learn which data is not important therefore can be forgotten or which data is important to keep.

#### What is the number of parameters in an LSTM cell?

The LSTM has a set of 2 matrices: $W_{h}$ and $W_{x}$ for each of the (3) gates (forget gate/input gate/output gate). Each $W_{h}$ has $U \times U$ elements and each $W_{x}$ has $F \times U$ elements. There is another set of these matrices for updating the cell state (new candidate). Similarly, $W_{xc}$ has $F \times U$ and $W_{hc}$ has $U \times U$ elements. On top of the mentioned matrices, you need to count the biases. Each bias for 3 gates and new candidate has $U$ elements. Hence total number parameters is $4(UF +  U^{2} + U)$.

#### Why stacking LSTM layers?

The main reason for stacking LSTM cells is to allow for greater model complexity. The addition of more layers increases the capacity of the network, making it capable of learning a large training dataset and efficiently representing more complex mapping functions from inputs to outputs. In case of a simple feed forward network, we stack layers to create a hierarchial feature representation of the input data to then use for some machine learning task. The same applies for stacked LSTMs. 

#### What is an autoencoder?

#### What are some limitations of deep learning?

#### What is transfer learning ?

A deep learning model trained on a specific task (a pre-trained model) can be reused for different problem in the same domain even if the amount of data is not that huge.

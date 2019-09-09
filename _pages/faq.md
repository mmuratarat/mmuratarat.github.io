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
15. What is Ax = b? When does Ax =b has a unique solution?}
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
31. What is eigendecomposition, eigenvectors and eigenvalues?
32. How to find eigenvalues of a matrix?
33. Write the eigendecomposition formula for a matrix. If the matrix is real symmetric, how will this change?
34. Is the Eigendecomposition guaranteed to be unique? If not, then how do we represent it?
35. What are positive definite, negative definite, positive semi definite and negative semi definite matrices?
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


[Set Theory](#set-theory)

1. [What is a random experiment?](#what-is-a-random-experiment)
2. [What is a sample space?](#what-is-a-sample-space)
3. [What is an empty set?](#what-is-an-empty-set)
4. [What is an event?](#what-is-an-event)
5. [What are the operations on a set?](#what-are-the-operations-on-a-set)
6. [What is mutually exclusive (disjoint) events?](#what-is-mutually-exclusive-disjoint-events)
7. [What is a non-disjoint event?](#what-is-a-non-disjoint-event)
7. [What is exhaustive events?](#what-is-exhaustive-events)
8. [What is Inclusion-Exlusive Principle?](#what-is-inclusion-exlusive-principle)


[Probability](#probability)

9. [What is a probability?](#what-is-a-probability)
10. [What are the probability axioms?](#what-are-the-probability-axioms)
11. [What is a random variable?](#what-is-a-random-variable)
12. Compare “Frequentist probability” vs. “Bayesian probability”?
13. What is a probability distribution?
14. What is a probability mass function?
15. What is a probability density function?
16. What is a joint probability distribution?
17. What are the conditions for a function to be a probability mass function?
18. What are the conditions for a function to be a probability density function?
19. What is a marginal probability? Given the joint probability function, how will you calculate it?
20. What is conditional probability? Given the joint probability function, how will you calculate it?
21. State the Chain rule of conditional probabilities.
22. What are the conditions for independence and conditional independence of two random variables?
23. [What are expectation, variance and covariance?](#what-are-expectation-variance-and-covariance)
24. What is the covariance for a vector of random variables?
25. [What is a Bernoulli distribution? Calculate the expectation and variance of a random variable that follows Bernoulli distribution?](#what-is-a-bernoulli-distribution-calculate-the-expectation-and-variance-of-a-random-variable-that-follows-bernoulli-distribution)
26. What is a multinoulli distribution?
27. What is a normal distribution?
28. Why is the normal distribution a default choice for a prior over a set of real numbers?
29. What is the central limit theorem?
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
40. What are Shannon entropy and differential entropy?
41. What is Kullback-Leibler (KL) divergence?
42. [Can KL divergence be used as a distance measure?](#can-kl-divergence-be-used-as-a-distance-measure)
43. Define cross-entropy.
44. What are structured probabilistic models or graphical models?
45. In the context of structured probabilistic models, what are directed and undirected models? How are they represented? What are cliques in undirected structured probabilistic models?
46. What is population mean and sample mean?
47. What is population standard deviation and sample standard deviation?
48. Why population s.d. has N degrees of freedom while sample s.d. has N-1 degrees of freedom? In other words, why 1/N inside root for pop. s.d. and 1/(N-1) inside root for sample s.d.?
49. What is the formula for calculating the s.d. of the sample mean?
50. [What is confidence interval?](#what-is-confidence-interval)
51. What is standard error?
52. [What is a p-value?](#what-is-a-p-value)



[General Machine Learning](#general-machine-learning)

1. What is an epoch, a batch and an iteration?
2. What is the matrix used to evaluate the predictive model? How do you evaluate the performance of a regression prediction model vs a classification prediction model?
3. What are the assumptions required for linear regression?


## Linear Algebra

#### What are scalars, vectors, matrices, and tensors?

Scalars are single numbers and are an example of a 0th-order tensor. The notation $x \in \mathbb{R}$ states that the scalar value $x$ is an element of (or member of) the set of real-valued numbers, $\mathbb{R}$.

There are various sets of numbers of interest within machine learning. $\matbb{N}$ represents the set of positive integers $(1,2,3, ...)$. $\matbb{Z}$ represents the integers, which include positive, negative and zero values. $\mathbb{Q}$ represents the set of rational numbers that may be expressed as a fraction of two integers.

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

The transpose of a matrix results from "flipping" the rows and columns. Given a matrix $A \in \mathbb{R}^{m \times n}$, its transpose, written as $A^{T} \in \mathbb{R}^{n \times m}$, is the matrix whose entries are given by $\left(A^{T}\right)_{ij} = A_{ji}$.

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

**Determine its rank**. In order for a square matrix $A$ to have an inverse $A^{-1}$, then $A$ must be full rank. The rank of a matrix is a unique number associated with a square matrix. If the rank of an $n \times n$ matrix is less than $n$, the matrix does not have an inverse. 

**Compute its determinant**. The determinant is another unique number associated with a square matrix. When the determinant for a square matrix is equal to zero, the inverse for that matrix does not exist.

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

#### What is the trace of a scalar?

A scalar is its own trace $a=Tr(a)$


## Numerical Optimization

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
Disjoint Events, by definition, can not happen at the same time. A synonym for this term is mutually exclusive. Non-disjoint events, on the other hand, can happen at the same time. For example, a student can get grade A in Statistics course and A in History course at the same time.

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/nondisjoint_events.png?raw=true)

For example, if events $A$ and $B$ are non-disjoint events, the probability of A or B happening (union of these events) is given by:

$$
P(A \cup B) = P(A) + P(B) - P(A \cap B)
$$

#### What is exhaustive events?
When two or more events form the sample space ($S$) collectively than it is known as collectively exhaustive events. The $n$ events $A_{1}, A_{2}, A_{3}, \ldots, A_{n}$ are said to be exhaustive if $A_{1} \cup A_{2} \cup A_{3} \cup \ldots \cup A_{n} = S$. 

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
A random variable is a variable whose values depend on all the possible outcomes of a natural phenomenon. There are two types of random variables, discrete and continuous. 

A discrete random variable is one which may take on only a countable number of distinct values such as 0,1,2,3,4,... Discrete random variables are usually (but not necessarily) counts. If a random variable can take only a finite number of distinct values, then it must be discrete. Examples of discrete random variables include the number of children in a family, the Friday night attendance at a cinema, the number of patients in a doctor's surgery, the number of defective light bulbs in a box of ten.

A continuous random variable is one which takes an infinite number of possible values. Continuous random variables are usually measurements. Examples include height, weight, the amount of sugar in an orange, the time required to run a mile.

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
corr(X, Y) = \rho_{X, Y}= \frac{cov(X,Y)}{\sigma_{X}\sigma_{y}}
$$

The covariance is especially useful when looking at the variance of the sum of two random variates, since

$$
Var(X+Y) = Var(X)+ Var(Y) + 2cov(X,Y)
$$

The covariance is symmetric by definition since

$$
cov(X,Y)=cov(Y,X). 
$$

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

Bernoulli distribution is a special case of Binomial distribution. If $X_{1},\dots ,X_{n}$ are independent, identically distributed (i.i.d.) random variables, all Bernoulli trials with success probability $p$, then their sum is distributed according to a binomial distribution with parameters $n$ and $p$:

$$
\sum _{k=1}^{n}X_{k}\sim Binomial(n,p)
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

#### Can KL divergence be used as a distance measure?
It may be tempting to think of KL Divergence as a distance metric, however we cannot use KL Divergence to measure the distance between two distributions. The reason for this is that KL Divergence is not symmetric, meaning that $D_{KL}(p\mid \mid q)$ may not be equal to $D_{KL}(q\mid \mid p)$.


#### What is confidence interval?

The purpose of taking a random sample from a population and computing a statistic, such as the mean from the data, is to approximate the mean of the population. How well the sample statistic estimates the underlying population value is always an issue. In statistical inference, one wishes to estimate population parameters using observed sample data. A confidence interval gives an estimated range of values which is likely to include an unknown population parameter, the estimated range being calculated from a given set of sample data

Confidence intervals are constructed at a confidence level, such as $95\%$, selected by the user. What does this mean? It means that if the same population is sampled on numerous occasions and interval estimates are made on each occasion, the resulting intervals would bracket the true population parameter in approximately $95\%$ of the cases.

For example, when we try to construct confidence interval for the true mean of heights of men, the "$95\%$" says that $95\%$ of experiments will include the true mean, but $5\%$ won't. So there is a 1-in-20 chance ($5\%$) that our confidence interval does NOT include the true mean. 

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

![](https://github.com/mmuratarat/mmuratarat.github.io/blob/master/_posts/images/type_I_type_II_errors.png?raw=true)

__Type I error__ is the rejection of a true null hypothesis (also known as a "false positive" finding or conclusion), while a __type II error__ is the non-rejection of a false null hypothesis (also known as a "false negative" finding or conclusion). 

When the null hypothesis is true and you reject it, you make a type I error. The probability of making a type I error is alpha, which is the level of significance you set for your hypothesis test.

The power of a test is one minus the probability of type II error (beta), which is the probability of rejecting of rejecting the null hypothesis when it is false. The power of a test tells us how likely we are to find a significant difference given that the alternative hypothesis is true (the true mean is different from the mean under the null hypothesis). Therefore, power should be maximized when selecting statistical methods. 

The chances of committing these two types of errors are inversely proportional—that is, decreasing Type I error rate increases Type II error rate, and vice versa. To decrease your chance of committing a Type I error, simply make your alpha value more stringent. To reduce your chance of committing a Type II error, increase your analyses’ power by either increasing your sample size or relaxing your alpha level!

## General Machine Learning

#### What is an epoch, a batch and an iteration?

A batch is the complete dataset. Its size is the total number of training examples in the available dataset.

Mini-batch size is the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.

A Mini-batch is a small part of the dataset of given mini-batch size.
One epoch = one forward pass and one backward pass of all the training examples

Number of iterations = An iteration describes the number of times a batch of data passed through the algorithm, number of passes, each pass using [batch size] number of examples. To be clear, one pass = one forward pass + one backward pass (we do not count the forward pass and backward pass as two different passes).

_Example_: if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch.

#### What is the matrix used to evaluate the predictive model? How do you evaluate the performance of a regression prediction model vs a classification prediction model?

Confusion Matrix, also known as an error matrix, describes the complete performance of the model. Each row of the matrix represents the instances in a predicted class while each column represents the instances in an actual class (or vice versa). The name stems from the fact that it makes it easy to see if the system is confusing two classes (i.e. commonly mislabeling one as another). It is a special kind of contingency table, with two dimensions ("actual" and "predicted"), and identical sets of "classes" in both dimensions (each combination of dimension and class is a variable in the contingency table).

* **Regression problems**: Mean Absolute Error, Mean Squared Error, Root Mean Squared Error, R-squared
* **Classification problems**: Accuracy, Precision, Recall, Sensitivity, Specificity, False Positive Rate, F1 Score, AUC, Lift and gain charts

* Linear Relationship between the features and target
* The number of observations must be greater than number of features
* No Multicollinearity between the features: Multicollinearity is a state of very high inter-correlations or inter-associations among the independent variables.It is therefore a type of disturbance in the data if present weakens the statistical power of the regression model. Pair plots and heatmaps(correlation matrix) can be used for identifying highly correlated features.
* Homoscedasticity of residuals or equal variance $Var \left(\varepsilon \mid X_{1} = x_{1}, \cdot X_{p}=x_{p} \right) = \sigma^{2}$: Homoscedasticity describes a situation in which the error term (that is, the "noise" or random disturbance in the relationship between the features and the target) is the same across all values of the independent variables. 
     More specifically, it is assumed that the error (a.k.a residual) of a regression model is homoscedastic across all values of the predicted value of the dependent variable. A scatter plot of residual values vs predicted values is a good way to check for homoscedasticity. There should be no clear pattern in the distribution and if there is a specific pattern, the data is heteroscedastic. 
* Normal distribution of error terms $\varepsilon \sim N(0, \sigma^{2})$: The fourth assumption is that the error(residuals) follow a normal distribution.However, a less widely known fact is that, as sample sizes increase, the normality assumption for the residuals is not needed. More precisely, if we consider repeated sampling from our population, for large sample sizes, the distribution (across repeated samples) of the ordinary least squares estimates of the regression coefficients follow a normal distribution. As a consequence, for moderate to large sample sizes, non-normality of residuals should not adversely affect the usual inferential procedures. This result is a consequence of an extremely important result in statistics, known as the central limit theorem.
     Normal distribution of the residuals can be validated by plotting a q-q plot.
* No autocorrelation of residuals (Independence of errors $E(\varepsilon_{i} \varepsilon_{j}] = 0, \,\,\, i \neq j$): Autocorrelation occurs when the residual errors are dependent on each other. The presence of correlation in error terms drastically reduces model's accuracy. This usually occurs in time series models where the next instant is dependent on previous instant.
    Autocorrelation can be tested with the help of Durbin-Watson test. The null hypothesis of the test is that there is no serial correlation. 

---
layout: post
title: "Derivation of Softmax Function"
author: "MMA"
comments: true
---

In [this post](https://mmuratarat.github.io/2019-01-14/Implementing-Softmax-Function){:target="_blank"}, we talked a little about softmax function and how to easily implement it in Python. Now, we will go a bit in details and to learn how to take its derivative since it is used pretty much in Backpropagation of a Neural Network.

Softmax function is given by:

$$ S(x_{i}) = \frac{e^{x_i}}{\sum_{k=1}^K e^{x_k}} \;\;\;\text{ for } i = 1, \dots, K$$

As one can easily see that the function takes K-dimensional vector of arbitrary real-valued scores (e.g., in the case of multi-class classification, this vector has length equal to the number of classes $K$) and produces another K-dimensional vector whose values are squashed between zero and one that add up to one. It maps $S(x): \mathbb{R}^{K} \rightarrow \mathbb{R}^{K}$. 

$$S(x): \begin{bmatrix}x_{1} \\ x_{2}\\ \vdots \\x_{K} \end{bmatrix} \rightarrow \begin{bmatrix}S_{1} \\ S_{2}\\ \vdots \\S_{K}\end{bmatrix} $$ 

Intiutively, the softmax function is a "soft" version of the maximum function. A "hardmax" function (i.e. argmax) is not differentiable. The softmax gives at least a minimal amount of probability to all elements in the output vector, and so is nicely differentiable. Instead of selecting one maximal element in the vector, the softmax function breaks the vector up into parts of a whole (1.0) with the maximal input element getting a proportionally larger chunk, but the other elements get some of it as well. Another nice property of it, the output of the softmax function can be interpreted as a probability distribution, which is very useful in Machine Learning because all the output values are in the range of (0,1) and sum up to $1.0$. This is especially useful in multi-class classification because we often want to assign probabilities that our instance belong to one of a set of output classes.

For example, let's consider we have 4 classes, i.e. $K=4$, and unscaled scores (logits) are given by $[2,4,2,1]$. The simple argmax function outputs $[0,1,0,0]$. The argmax is the goal, but it's not differentiable and we can't train our model with it. A simple normalization, which is differentiable, outputs the following probabilities $[0.2222,0.4444,0.2222,0.1111]$. That's really far from the argmax! Whereas the softmax outputs $[0.1025,0.7573,0.1025,0.0377]$. That's much closer to the argmax! Because we use the natural exponential, we hugely increase the probability of the biggest score and decrease the probability of the lower scores when compared with standard normalization. Hence the "max" in softmax.

Softmax is fundamentally a vector function. It takes a vector as input and produces a vector as output. In other words, it has multiple inputs and outputs. Therefore, when we try to find the derivative of the softmax function, we talk about a Jacobian matrix, which is the matrix of all first-order partial derivatives of a vector-valued function.

Since softmax is a $\mathbb{R}^{K} \rightarrow \mathbb{R}^{K}$ mapping function, the most general Jacobian matrix for it is:

$$\frac{\partial S}{\partial x} =
\begin{bmatrix}
\frac{\partial S_{1}}{\partial x_{1}} & \frac{\partial S_{1}}{\partial x_{2}} & \cdots & \frac{\partial S_{1}}{\partial x_{K}} \\
\frac{\partial S_{2}}{\partial x_{1}} & \frac{\partial S_{1}}{\partial x_{2}} & \cdots & \frac{\partial S_{K}}{\partial x_{K}} \\
\vdots & \vdots & \cdots & \vdots \\
\frac{\partial S_{K}}{\partial x_{1}} & \frac{\partial S_{K}}{\partial x_{2}} & \cdots & \frac{\partial S_{K}}{\partial x_{K}} \\
\end{bmatrix}
$$

Let's compute $\frac{\partial S_{i}}{\partial x_{j}}$ for some arbitrary $i$ and $j$:

$$
\frac{\partial S_{i}}{\partial x_{j}} = \dfrac{\partial}{\partial x_{j}}\frac{e^{x_{i}}}{\sum_{k=1}^{K} e^{x_k}} 
$$

By the quotient rule for derivatives, for $f(x) = \dfrac{g(x)}{h(x)}$, the derivative of $f(x)$ is given by:

$$f^{'}(x) = \frac{g^{'}(x)h(x) - h^{'}(x)g(x)}{\left[h(x) \right]^{2}}$$

In our case, $g_{i} = e^{x_{i}}$ and $h_{i} = \sum_{k=1}^{K} e^{x_k}$.

No matter which $x_{j}$, when we compute the derivative of $h_{i}$ with respect to $x_{j}$, the answer will always be $e^{x_{j}}$.

$$
\frac{\partial}{\partial x_{j}} h_{i} = \frac{\partial}{\partial x_{j}} \sum_{k=1}^{K} e^{x_k} = \sum_{k=1}^{K}  \frac{\partial}{\partial x_{j}} e^{x_k} = e^{x_{j}}
$$

because, $\frac{\partial}{\partial x_{j}} e^{x_k} = 0$ for $k \neq j$. Remember that $x_{k}$ is $k$-th element in the vector $x$.

However, this is not the case for $g_{i}$.

Derivative of $g_{i}$ with respect to $x_{j}$ is $e^{x_{j}}$ only if $i = j$ because only then, $g_{i}$ has $x_{j}$ anywhere in it. Otherwise, it is a constant and its derivative is 0. 


Therefore, if we derive the gradient of the diagonal of the Jacobian matrix, which is to say that $i = j$, we will have

\begin{equation}
\begin{split}
\dfrac{\partial \frac{e^{x_{i}}}{\sum_{k=1}^{K} e^{x_k}} }{\partial x_{j}} &= \frac{e^{x_{i}} \sum_{k=1}^{K} e^{x_k}- e^{x_{j}}e^{x_{i}}}{\left[\sum_{k=1}^{K} e^{x_k} \right]^{2}}\\
&=\frac{e^{x_{i}}\left(\sum_{k=1}^{K} e^{x_k} - e^{x_{j}}\right)}{\left[\sum_{k=1}^{K} e^{x_k} \right]^{2}}\\
&=S_{i} (1-S_{j})
\end{split}
\end{equation}

Similarly, deriving the off diagonal entries of the Jacobian matrix will yield:

\begin{equation}
\begin{split}
\dfrac{\partial \frac{e^{x_{i}}}{\sum_{k=1}^{K} e^{x_k}} }{\partial x_{j}} &= \frac{0 \sum_{k=1}^{K} e^{x_k}- e^{x_{j}}e^{x_{i}}}{\left[\sum_{k=1}^{K} e^{x_k} \right]^{2}}\\
&=-\frac{e^{x_{j}}}{\sum_{k=1}^{K} e^{x_k}} \frac{e^{x_{i}}}{\sum_{k=1}^{K} e^{x_k}}
&=-S_{j}S_{i}
\end{split}
\end{equation}

If we summarize:

\begin{equation}
\frac{\partial S_{i}}{\partial x_{j}} = \left \\{ \begin{array}{ll}
         S_{i} (1-S_{j}) & \mbox{if $i = j$};\\
        -S_{j}S_{i} & \mbox{if $i \neq j$}.\end{array}  \right.
\end{equation}

Sometimes, this piecewise function can be put together using [Kronecker delta function](https://en.wikipedia.org/wiki/Kronecker_delta){:target="_blank"}:

\begin{equation}
\frac{\partial S_{i}}{\partial x_{j}} = S_{i} (\delta_{ij}-S_{j})
\end{equation}

where,

\begin{equation}
\delta_{ij} = \left \\{ \begin{matrix}{ll}
         1 & \mbox{if $i = j$};\\
         0 & \mbox{if $i \neq j$}.\end{matrix}  \right.
\end{equation}

# REFERENCES:
1. [http://cs231n.github.io/linear-classify/#softmax](http://cs231n.github.io/linear-classify/#softmax){:target="_blank"}
2. [https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/){:target="_blank"}
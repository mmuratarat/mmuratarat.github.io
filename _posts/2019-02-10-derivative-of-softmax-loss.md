---
layout: post
title: "Derivative of Loss Function w.r.t. softmax function"
author: "MMA"
comments: true
---

Softmax function is given by:

$$ S(x_{i}) = \frac{e^{x_i}}{\sum_{k=1}^K e^{x_k}} \;\;\;\text{ for } i = 1, \dots, K$$

Softmax is fundamentally a vector function. It takes a vector as input and produces a vector as output. In other words, it has multiple inputs and outputs. Therefore, when we try to find the derivative of the softmax function, we talk about a Jacobian matrix, which is the matrix of all first-order partial derivatives of a vector-valued function. More details are provided [here](https://mmuratarat.github.io/2019-01-27/derivation-of-softmax-function){:target="_blank"}.

Multi-class cross entropy formula is as follows:

$$ L = - \frac{1}{n} \sum_{i=1}^{n}  \sum_{j=1}^{K} \left[y_{ij} \log (p_{ij}) \right]$$

where $i$ indexes samples/observations and $j$ indexes classes. Here, $y_{ij}$ and $p_{ij}$ are expected to be probability distributions over $K$ classes. In a neural network, $y_{ij}$ is one-hot encoded labels and $p_{ij}$ is scaled (softmax) logits. More details are provided [here](https://mmuratarat.github.io/2018-12-21/cross-entropy){:target="_blank"}.

The loss for a single istance can be computed by,

$$ L = - \sum_{j} \left[y_{j} \log (p_{j}) \right]$$

Most of the time, $p_{j} = softmax(o_{i})$ where $o_{i}$ is the unscaled score (logit), which is also a vector. 

Since we know the derivative of softmax function with respect to its vector input, we can compute the derivative of the loss with respect to unscaled logit vector $o_{i}$. We have two options here: $i = j$ and $i \neq j$.

$$
\begin{split}
\frac{\partial}{\partial o_{i}} L &= - \sum_{j=i}y_{j} \frac{\partial \log (p_{j})}{\partial o_{i}} - \sum_{j \neq i}y_{j}\frac{\partial \log (p_{j})}{\partial o_{i}}\\
&= - \sum_{j=i}y_{j} \frac{1}{p_{j}}\frac{\partial p_{j}}{\partial o_{i}} - \sum_{j \neq i}y_{j}\frac{1}{p_{j}}\frac{\partial p_{j}}{\partial o_{i}}\\
&= - y_{i} \frac{1}{p_{j}} p_{j}(1-p_{i}) - \sum_{j \neq i}y_{j}\frac{1}{p_{j}}(-p_{j}p_{i})\\
& = - y_{i} (1- p_{i}) - \sum_{j \neq i}y_{j}\frac{1}{p_{j}}(-p_{j}p_{i})\\
&=- y_{i} + \underbrace{y_{i}p_{i} + \sum_{j \neq i}y_{j}p_{i}}_{ \bigstar } \\
&= p_{i}\left(\sum_{j}y_{j} \right) - y_{i} = p_{i} - y_{i}
\end{split}
$$

given that $\sum_{j}y_{j} = 1$ because $y$ is one-hot encoded vector.

Here, one needs to remember $\frac{\partial \log{f(x)}}{\partial x}=\frac{1}{f(x)}\frac{\partial f(x)}{\partial x}$, $\frac{\partial p_j}{\partial o_i} = p_i(1 - p_i),\quad i = j$ and $\frac{\partial p_j}{\partial o_i} = -p_i p_j,\quad i \neq j$, which the last two equations are nothing but derivative of softmax function with respect to logits. which can be found [here](https://mmuratarat.github.io/2019-01-27/derivation-of-softmax-function){:target="_blank"}.

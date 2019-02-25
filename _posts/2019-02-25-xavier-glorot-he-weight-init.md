---
layout: post
title: "Weight Initialization Schemes - Xavier (Glorot) and He"
author: "MMA"
comments: true
---


When you are working with deep neural networks, initializing the network with the right weights can be the hard to deal with. 

## Normal Distribution
We don’t know anything about the data, so we are not sure how to assign the weights that would work in that particular case. One good way is to assign the weights from a Gaussian distribution. Obviously this distribution would have zero mean and some finite variance. 

Suppose we have an inout $X$ with $n$ components and a linear neuron with random weights $W$ that spits out an output $Y$. The variance of $y$ can be written as:

$$
Y = W_{1}X_{1} + W_{2}X_{2} + \cdots + W_{n}X_{n}
$$

We know that the variance of $W_{i}X_{i}$ is

$$
Var(W_{i}X_{i}) =  E(X_{i})^{2} Var(W_{i}) + E(W_{i})^{2} Var(X_{i}) +  Var(W_{i})Var(X_{i})
$$

Here, we assume that $X_{i}$ and $W_{i}$ are all identically and independently distributed (Gaussian distribution with zero mean), we can work out the variance of $Y$ which is:

$$
\begin{split}
Var(Y) &= Var(W_{1}X_{1} + W_{2}X_{2} + \cdots + W_{n}X_{n}) \\
&= Var(W_{1}X_{1}) + Var(W_{2}X_{2}) + \cdots + Var(W_{n}X_{n})\\
&= nVar(W_{i})Var(X_{i})
\end{split}
$$

The variance of the output is the variance of the input but it is scaled by $nVar(W_{i})$. Hence, if we want the variance of $Y$ to be equal to the variance of $X$, then the term $nVar(W_{i})$ should be equal to 1. Hence, the variance of the weight should be:

$$
Var(W_{i}) = \frac{1}{n} = \frac{1}{n_{in}}
$$

This is Xavier Initialization formula. We need to pick the weights from a Gaussian distribution with zero mean and a variance of $\frac{1}{n_{in}}$ where $n_{in}$ is the number of input neurons. That is how Xavier (Glorot) initialization is implemented in Caffee library. 

Similarly, if we go through backpropagation, we apply the same steps and get:

$$
Var(W_{i}) = \frac{1}{n_{out}}
$$

In order to keep the variance of the input and the output gradient the same, these two constraints can only be satisfied simultaneously if $n_{in} = n_{out}$. However, in the general case, the $n_{in}$ and $n_{out}$ of a layer may not be equal, and so as a sort of compromise, Glorot and Bengio suggest using the average of the $n_{in}$ and $n_{out}$, proposing that:


$$
Var(W_{i}) = \frac{1}{n_{avg}}
$$

where $n_{avg} = \frac{n_{in} + n_{out}}{2}$.

Note that when the number of input connections is roughly equal to the number of output connections, you get the simpler equations:

$$
Var(W_{i}) = \frac{1}{n_{in}}
$$

## Uniform Distribution
What if we want to use a Uniform distribution? 

If sampling from a uniform distribution, this translates to sampling the interval $[−r,r]$, where $r = \sqrt{\frac{6}{n_{in} + n_{out}}}$. The weird-looking $\sqrt{6}$ factor comes from the fact that the variance of a uniform distribution over the interval $[−r,r]$ is $r^{2}/3$. So for forward pass, if we want the variance of weights to be equal to $\frac{1}{n_{in}}$, we need to choose $r = \sqrt{\frac{3}{n_{in}}}$. If you go through the same steps for the backpropagated signal and use Glorot and Bengio implementation to reconcile Forward and Backward Passes, we will have:

$$
r = \sqrt{\frac{6}{n_{in} + n_{out}}}
$$

Similarly, note that when the number of input connections is roughly equal to the number of output connections, you get the simpler equations:

$$
r = \frac{\sqrt{3}}{\sqrt{n_{in}}}
$$

# He Initialization
Glorot and Bengio considered logistic sigmoid activation function, which was the default choice at that moment for their weight initialization scheme. Later on, the sigmoid activation was surpassed by ReLu, because it allowed to solve vanishing / exploding gradients problem. However, it turns out  Xavier (Glorot) Initialization isn’t quite as optimal for ReLU functions. Consequently, there appeared a new initialization technique, which applied the same idea (balancing of the variance of the activation) to this new activation function and now it often referred to as He initialization. The initialization strategy for ReLU activation function and its variants is sometimes called He initialization. There is only one tiny adjustment we need to make, which is to multiply the variance of the weights by 2! 

Some papers in the literature have provided similar strategies for different activation functions which is shown below:

| Activation Function         	| Uniform Distribution [-r, +r]                   	| Normal Distribution                             	|
|-----------------------------	|-------------------------------------------------	|-------------------------------------------------	|
| Sigmoid Function            	| $r = \sqrt{\frac{6}{n_{in} + n_{out}}}$         	| $r = \sqrt{\frac{2}{n_{in} + n_{out}}}$         	|
| Hyperbolic Tangent Function 	| $r = 4\sqrt{\frac{6}{n_{in} + n_{out}}}$        	| $r = 4\sqrt{\frac{2}{n_{in} + n_{out}}}$        	|
| ReLU (and its variants)     	| $r = \sqrt{2}\sqrt{\frac{6}{n_{in} + n_{out}}}$ 	| $r = \sqrt{2}\sqrt{\frac{2}{n_{in} + n_{out}}}$ 	|

# Tensorflow Implementation
In Tensorflow, He initialization is implemented in [variance_scaling_initializer()](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling){:target="_blank"} function (which is, in fact, a more general initializer, but by default performs He initialization), while Xavier initializer is logically [xavier_initializer()](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/xavier_initializer){:target="_blank"}.

# REFERENCES
1. [Understanding the difficulty of training deep feedforward neural networks, Xavier Glorot, Yoshua Bengio ; PMLR 9:249–256](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf){:target="_blank"}
2.[https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404](https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404){:target="_blank"}
3. [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification by Kaiming He,  Xiangyu Zhang, Shaoqing Ren and Jian Sun](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf){:target="_blank"}
4. [http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization](http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization){:target="_blank"}

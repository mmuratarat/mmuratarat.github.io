---
layout: post
title: "Backpropagation Through Time for Recurrent Neural Network"
author: "MMA"
comments: true
---

![](https://raw.githubusercontent.com/mmuratarat/mmuratarat.github.io/master/_posts/images/BPTT.png)



The dynamical system is defined by:

$$
\begin{split}
    h_{t} & = f_{h} (X_{t}, h_{t-1})\\
    \hat{y}_{t} &= f_{o}(h_{t})
\end{split}
$$

A conventional RNN is constructed by defining the transition function and the output function for a single instance:

$$
\begin{split}
    h_{t} & = f_{h} (X_{t}, h_{t-1}) = \phi_{h}(W_{xh}^{T} \cdot X_{t} + W_{hh}^{T}\cdot h_{t-1} +b_{h})\\
    \hat{y}_{t} &= f_{o}(h_{t}) = \phi_{o}(W_{yh}^{T}\cdot h_{t} + b_{y})
\end{split}
$$

where $W_{xh}$, $W_{hh}$ and $W_{yh}$ are weight matrices for the input, reccurent connections, and the output,  respectively and $\phi_{h}$ and $\phi_{o}$ are element-wise nonlinear functions. It is usual to use a saturating nonlinear function such as logistic sigmoid function or a hyperbolic tangent function for $\phi_{h}$. $\phi_{o}$  is generally softmax activation for classification problem. 

**NOTE**: Reusing same weight matrix every time step! $W$ is shared across time - reduces the number of parameters!

Just like for feedforward neural networks, we can compute a recurrent layer’s output in one shot for a whole mini-batch by placing all the inputs at time step $t$ in an input matrix $X_{t}$:

$$
\begin{split}
    h_{t} & = tanh(X_{t}\cdot W_{xh} + h_{t-1}\cdot  W_{hh} + b_{h})\\
    &= \phi_{h}( [X_{t} h_{t-1}] \cdot W + b_{h})\\
    o_{t} &= h_{t}\cdot W_{yh} + b_{y}\\
    \hat{y}_{t} &= softmax(o_{t})
\end{split}
$$

1. The weight matrices $W_{xh}$ and $W_{yh}$ are often concatenated vertically into a single weight matrix $W$ of shape $(n_{inputs} +  n_{neurons}) \times  n_{neurons}$.
2. The notation $[X_{t} h_{t-1}]$ represents the horizontal concatenation of the matrices $X_{t}$ and $h_{t-1}$, shape of $m \times (n_{inputs} + n_{neurons})$  

Let's denote $m$ as the number of instances in the mini-batch, $n_{neurons}$ as the number of neurons, and $n_{inputs}$ as the number of input features.

1. $X_{t}$ is an $m \times n_{inputs}$ matrix containing the inputs for all instances.
2. $h_{t-1}$ is an $m \times n_{neurons}$ matrix containing the hidden state of the previous time-step for all instances.
3. $W_{xh}$ is an $n_{inputs} \times n_{neurons}$ matrix containing the connection weights between input and the hidden layer.
4. $W_{hh}$ is an $n_{neurons} \times n_{neurons}$ matrix containing the connection weights between two hidden layers.
5. $W_{yh}$ is an $n_{neurons} \times n_{neurons}$ matrix containing the connection weights between the hidden layer and the output.
6. $b_{h}$ is a vector of size $n_{neurons}$ containing each neuron’s bias term.
7. $b_{y}$ is a vector of size $n_{neurons}$ containing each output’s bias term.
8. $y_{t}$ is an $m \times n_{neurons}$ matrix containing the layer’s outputs at time step $t$ for each instance in the mini-batch 

# Backpropagation Through Time
In order to do backpropagation through time to train an RNN, we need to compute the loss function first:

$$
\begin{split}
L (\hat{y}, y) & = \sum_{t = 1}^{T} L_{t}(\hat{y}_{t}, y_{t}) \\
& = - \sum_{t = 1}^{T} y_{t} \log \hat{y}_{t} \\
& = -\sum_{t = 1}^{T}y_{t} log \left[softmax(o_{t}) \right]
\end{split}
$$

Note that the weight $W_{yh}$ is shared across all the time sequence. Therefore, we can differentiate to it at the each time step and sum all together:

$$
\frac{\partial L}{\partial W_{yh}} = \sum_{t = 1}^{T} \frac{\partial L_{t}}{\partial W_{yh}}  = \sum_{t = 1}^{T} \frac{\partial L_{t}}{\partial \hat{y}_{t}} \frac{\partial \hat{y}_{t}}{\partial o_{t}} \frac{\partial o_{t}}{\partial W_{yh}}
$$

Similarly, we can get the gradient w.r.t. bias $b_{y}$:

$$
\frac{\partial L}{\partial b_{y}}  = \sum_{t = 1}^{T} \frac{\partial L_{t}}{\partial \hat{y}_{t}} \frac{\partial \hat{y}_{t}}{\partial o_{t}} \frac{\partial o_{t}}{\partial b_{y}}
$$

Further, let's use $L_{t+1}$ to denote the output of the time-step $t+1$, $L_{t+1} = -y_{t+1} log \hat{y}_{t+1}$.

Now, let's go throught the details to derive the gradient with respect to $W_{hh}$, considering at the time step $t \rightarrow t+1$ (from time-step $t$ to $t+1$).

$$
\frac{\partial L_{t+1}}{\partial W_{hh}} =  \frac{\partial L_{t+1}}{\partial \hat{y}_{t+1}} \frac{\partial \hat{y}_{t+1}}{\partial h_{t+1}} \frac{\partial h_{t+1}}{\partial W_{hh}}
$$

where we consider only one time-step ($t \rightarrow t+1$). But, the hidden state $h_{t+1}$ partially depends also on $h_{t}$ according to the recursive formulation ($h_{t} = tanh(W_{xh}^{T} \cdot X_{t} + W_{hh}^{T}\cdot h_{t-1} +b_{h})$). Thus, at the time-step $t-1 \rightarrow t$, we can further get the partial derivative with respect to $W_{hh}$ as the following:

$$
\frac{\partial L_{t+1}}{\partial W_{hh}} =  \frac{\partial L_{t+1}}{\partial \hat{y}_{t+1}} \frac{\partial \hat{y}_{t+1}}{\partial h_{t+1}}\frac{\partial h_{t+1}}{\partial h_{t}} \frac{\partial h_{t}}{\partial W_{hh}}
$$

Thus, at the time-step $t+1$, we can compute the gradient and further use backpropagation through time from $t+1$ to $1$ to compute the overall gradient with respect to $W_{hh}$:

$$
\frac{\partial L_{t+1}}{\partial W_{hh}} = \sum_{k=1}^{t+1} \frac{\partial L_{t+1}}{\partial \hat{y}_{t+1}} \frac{\partial \hat{y}_{t+1}}{\partial h_{t+1}}\frac{\partial h_{t+1}}{\partial h_{k}} \frac{\partial h_{k}}{\partial W_{hh}}
$$

Note that $\frac{\partial h_{t+1}}{\partial h_{k}}$ is a chain rule in itself!  For example, $\frac{\partial h_{3}}{\partial h_{1}} = \frac{\partial h_{3}}{\partial h_{2}}\frac{\partial h_{2}}{\partial h_{1}}$. Also note that because we are taking the derivative of a vector function with respect to a vector, the result is a matrix (called the Jacobian matrix) whose elements are all the pointwise derivatives. We can rewrite the above gradient:

$$
\frac{\partial L_{t+1}}{\partial W_{hh}} = \sum_{k=1}^{t+1} \frac{\partial L_{t+1}}{\partial \hat{y}_{t+1}} \frac{\partial \hat{y}_{t+1}}{\partial h_{t+1}}  \left( \prod_{j = k} ^{t} \frac{\partial h_{j+1}}{\partial h_{j}} \right) \frac{\partial h_{k}}{\partial W_{hh}}
$$

where

$$
\prod^{t}_{j=k} \frac{\partial h_{j+1}}{\partial h_{j}} = \frac{\partial h_{t+1}}{\partial h_k} = \frac{\partial h_{t+1}}{\partial h_{t}}\frac{\partial h_{t}}{\partial h_{t-1}}...\frac{\partial h_{k+1}}{\partial h_k} 
$$

Aggregate the gradients with respect to $W_{hh}$ over the whote time-sequence with backpropagation, we can finally yield the following gradient with respect to $W_{hh}$:

$$
\frac{\partial L}{\partial W_{hh}} = \sum_{t}^{T} \sum_{k=1}^{t+1} \frac{\partial L_{t+1}}{\partial \hat{y}_{t+1}} \frac{\partial \hat{y}_{t+1}}{\partial h_{t+1}} \frac{\partial h_{t+1}}{\partial h_{k}}\frac{\partial h_{k}}{\partial W_{hh}}
$$

Now, let's work on to derive the gradient with respect to $W_{xh}$. Similarly, we consider the time-step $t+1$ (which gets only contribution from $X_{t+1}$) and calculate the gradients with respect to $W_{xh}$ as follows:

$$
\frac{\partial L_{t+1}}{\partial W_{xh}} = \frac{\partial L_{t+1}}{\partial \hat{y}_{t+1}} \frac{\partial \hat{y}_{t+1}}{\partial h_{t+1}}\frac{\partial h_{t+1}}{\partial W_{xh}} 
$$

Because $h_{t}$ and $X_{t+1}$ both make contribution to $h_{t+1}$, we need to backpropagate to $h_{t}$ as well.

If we consider the contribution from the time-step, we can further get:

$$
\frac{\partial L_{t+1}}{\partial W_{xh}} = \frac{\partial L_{t+1}}{\partial \hat{y}_{t+1}} \frac{\partial \hat{y}_{t+1}}{\partial h_{t+1}}\frac{\partial h_{t+1}}{\partial W_{xh}}  + \frac{\partial L_{t+1}}{\partial \hat{y}_{t+1}} \frac{\partial \hat{y}_{t+1}}{\partial h_{t+1}}\frac{\partial h_{t+1}}{\partial  h_{t}} \frac{\partial h_{t}}{\partial W_{xh}} 
$$

Thus, summing up all the contributions from $t+1$ to $1$ via Backpropagation, we can yield the gradient at the time-step $t+1$:

$$
\frac{\partial L_{t+1}}{\partial W_{xh}} = \sum_{k=1}^{t+1} \frac{\partial L_{t+1}}{\partial \hat{y}_{t+1}} \frac{\partial \hat{y}_{t+1}}{\partial h_{t+1}}\frac{\partial h_{t+1}}{\partial h_{k}} \frac{\partial h_{k}}{\partial W_{xh}} 
$$

Further, we can take the derivative with respect to $W_{xh}$ over the whole sequence as :

$$
\frac{\partial L} {\partial W_{xh}} = \sum_{t}^{T} \sum_{k=1}^{t+1} \frac{\partial L_{t+1}}{\partial \hat{y}_{t+1}} \frac{\partial \hat{y}_{t+1}}{\partial h_{t+1}} \frac{\partial h_{t+1}}{\partial h_{k}} \frac{\partial h_{k}}{\partial W_{xh}} 
$$

Do not forget that $\frac{\partial h_{t+1}}{\partial h_{k}}$ is a chain rule in itself, again!

# Vanishing/Exploding Gradients with vanilla RNNs

There are two factors that affect the magnitude of gradients - the weights and the activation functions (or more precisely, their derivatives) that the gradient passes through. In vanilla RNNs, vanishing/exploding gradient comes from the repeated application of the recurrent connections. More explicitly, they happen because of recursive derivative we need to compute $\frac{\partial h_{t+1}}{\partial h_k}$:

$$
\prod^{t}_{j=k} \frac{\partial h_{j+1}}{\partial h_{j}} = \frac{\partial h_{t+1}}{\partial h_k} = \frac{\partial h_{t+1}}{\partial h_{t}}\frac{\partial h_{t}}{\partial h_{t-1}}...\frac{\partial h_{k+1}}{\partial h_k} 
$$

Now let us look at a single one of these terms by taking the derivative of $h_{j+1}$ with respect to $h_{j}$ where diag turns a vector into a diagonal matrix because this recursive partial derivative is a Jacobian matrix:

$$
\frac{\partial h_{j+1}}{\partial h_{j}} =  diag(\phi_{h}^{\prime}(W_{xh}^{T} \cdot X_{j+1} + W_{hh}^{T}\cdot h_{j} +b_{h}))W_{hh}
$$

Thus, if we want to backpropagate through $t-k$ timesteps, this gradient will be :

$$
\prod^{t}_{j=k} \frac{\partial h_{j+1}}{\partial h_{j}} = \prod^{t}_{j=k} diag(\phi_{h}^{\prime}(W_{xh}^{T} \cdot X_{j+1} + W_{hh}^{T}\cdot h_{j} +b_{h}))W_{hh}
$$

If we perform eigendecomposition on the Jacobian matrix $\frac{\partial h_{j+1}}{\partial h_{j}}$, we get the eigenvalues $\lambda_{1}, \lambda_{2}, \cdots, \lambda_{n}$ where $\lvert\lambda_{1}\rvert \gt \lvert\lambda_{2}\rvert \gt\cdots \gt \lvert\lambda_{n}\rvert$ and the corresponding eigenvectors $v_{1}, v_{1},\cdots,v_{n}$.

Any change on the hidden state $\delta h_{j+1}$ in the direction of a vector $v_{i}$ has the effect of multiplying the change with the eigenvalue associated with this eigenvector i.e $\lambda_{i}\delta h_{j+1}$.

The product of these Jacobians implies that subsequent time steps, will result in scaling the change with a factor equivalent to λti.


As shown in [this paper](https://arxiv.org/pdf/1211.5063.pdf){:target="_blank"}, if the dominant eigenvalue of the matrix $W_{hh}$ is greater than 1, the gradient explodes. If it is less than 1, the gradient vanishes. The fact that this equation leads to either vanishing or exploding gradients should make intuitive sense. Note that the values of $\phi_{h}^{\prime}$ will always be less than 1. Because in vanilla RNN, the activation function  $\phi_{h}$ is used to be hyperbolic tangent whose derivative is at most $0.25$. So if the magnitude of the values of $W_{hh}$ are too small, then inevitably the derivative will go to 0. The repeated multiplications of values less than one would overpower the repeated multiplications of $W_{hh}$. On the contrary, make $W_{hh}$ too big and the derivative will go to infinity since the exponentiation of $W_{hh}$ will overpower the repeated multiplication of the values less than 1. In practice, the vanishing gradient is more common.



These problems ultimately shows that if the gradient vanishes it means the earlier hidden states have no real effect on the later hidden states, meaning no long term dependencies are learned! 



**Note**: LSTM does not protect you from exploding gradients! Therefore, successful LSTM applications typically use gradient clipping.


# REFERENCES
1. [http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/){:target="_blank"}
2. [https://arxiv.org/abs/1610.02583](https://arxiv.org/abs/1610.02583){:target="_blank"}
3. [https://github.com/go2carter/nn-learn/blob/master/grad-deriv-tex/rnn-grad-deriv.pdf](https://github.com/go2carter/nn-learn/blob/master/grad-deriv-tex/rnn-grad-deriv.pdf){:target="_blank"}
4. [http://willwolf.io/2016/10/18/recurrent-neural-network-gradients-and-lessons-learned-therein/](http://willwolf.io/2016/10/18/recurrent-neural-network-gradients-and-lessons-learned-therein/){:target="_blank"}
5. [https://weberna.github.io/blog/2017/11/15/LSTM-Vanishing-Gradients.html](https://weberna.github.io/blog/2017/11/15/LSTM-Vanishing-Gradients.html){:target="_blank"}
6. [https://medium.com/datadriveninvestor/how-do-lstm-networks-solve-the-problem-of-vanishing-gradients-a6784971a577](https://medium.com/datadriveninvestor/how-do-lstm-networks-solve-the-problem-of-vanishing-gradients-a6784971a577){:target="_blank"}
7. [https://arxiv.org/abs/1211.5063](https://arxiv.org/abs/1211.5063){:target="_blank"}
8. [https://www.jefkine.com/general/2018/05/21/2018-05-21-vanishing-and-exploding-gradient-problems/](https://www.jefkine.com/general/2018/05/21/2018-05-21-vanishing-and-exploding-gradient-problems/){:target="_blank"}
